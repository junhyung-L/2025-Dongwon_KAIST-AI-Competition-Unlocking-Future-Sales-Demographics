# -*- coding: utf-8 -*-
"""
Integrated pipeline with persona caching:
- (옵션) 네이버 가격 수집 → category_prices.csv
- (옵션) LightGBM 가격모델 학습 → price_model.pkl
- Persona-based Monthly Demand Forecast → submission.csv 외 산출물
- (신규) LLM 페르소나 JSON 캐시 저장/재사용 → 재현성 보장

Run examples (CLI):
  1) 가격 수집:         python forecast_pipeline.py gather_prices --product_csv product_info.csv
  2) 가격모델 학습:     python forecast_pipeline.py train_price_model --prices_csv category_prices.csv
  3) 제출파일 생성:     python forecast_pipeline.py make_submission --product_csv product_info.csv --use_llm 0 --mc_runs 7
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import os, re, json, random, math, datetime as dt, hashlib
import numpy as np, pandas as pd

# ---------- Optional deps (안 깔려도 동작하도록) ----------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdfcanvas
    _HAS_RL = True
except Exception:
    _HAS_RL = False

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# ---------- Dates / RNG ----------
FORECAST_START = dt.date(2024,7,1)
FORECAST_END   = dt.date(2025,6,30)
MONTHS = pd.period_range(FORECAST_START, FORECAST_END, freq='M')
RNG = np.random.default_rng(42)
random.seed(42)

# ---------- Sources registry ----------
class SourcesRegistry:
    def __init__(self): self._items=[]
    def add(self, kind, title, how_used, url=None, notes=""):
        self._items.append({"kind":kind,"title":title,"url":url,"how_used":how_used,
                            "notes":notes,"added_at":dt.datetime.now().isoformat()})
    def to_json(self, path="sources_used.json"):
        with open(path,"w",encoding="utf-8") as f: json.dump(self._items,f,ensure_ascii=False,indent=2)

SOURCES = SourcesRegistry()
SOURCES.add('paper','Using LLMs for Market Research (Brand, Israeli, Ngwe, 2024)',
            'LLM-as-simulator 절차/싱글턴 프롬프트 설계','(local pdf)')

# ---------- Persona schema ----------
ATTRIBUTES = [
    'age','gender','income_band','region','household_size','lifestyle',
    'health_focus','price_sensitivity','brand_loyalty','online_offline_mix',
    'channel_preference','promo_reactivity','ad_reach_susceptibility',
    'environmental_concern','innovation_seeker'
]

@dataclass
class Persona:
    persona_id:str; name:str
    age:int; gender:str; income_band:str; region:str; household_size:int; lifestyle:str
    health_focus:float; price_sensitivity:float; brand_loyalty:float; online_offline_mix:float
    channel_preference:str; promo_reactivity:float; ad_reach_susceptibility:float
    environmental_concern:float; innovation_seeker:float
    weights:Dict[str,float]; monthly_pattern:List[float]

# ---------- LLM prompt ----------
SINGLE_TURN_PERSONA_PROMPT = """
You are a market-simulation engine. Generate N synthetic Korean consumer personas for a **Dongwon new product** launch.
Return **valid JSON** only (list of persona objects).
Context:
- Category: {category}
- Product concept: {concept}
- Target price (KRW): {price}
- Packaging/size: {pack}
- Channels: {channels}
- Competitors: {competitors}
- Target market size (12-month addressable): {market_size}
- Launch months: 2024-07 to 2025-06

Each persona fields:
- persona_id, name
- age (18-69), gender ("남"|"여"), region (서울/수도권/광역시/기타), household_size (1-5)
- income_band ("~2천","2-4천","4-7천","7천~"), lifestyle (≤12 chars)
- health_focus, price_sensitivity, brand_loyalty, online_offline_mix, promo_reactivity,
  ad_reach_susceptibility, environmental_concern, innovation_seeker (0~1)
- channel_preference in {channel_vocab}
- weights: map **≥10** of {attribute_list} to weights in [-2.0, +2.0] (utility contribution for THIS product)
- monthly_pattern: list[12] for 2024-07..2025-06 in 0.6~1.4 (persona seasonality)

Constraints: diversify personas; JSON only (no comments/trailing commas).
""".strip()

CHANNELS_VOCAB = ['hypermarket','convenience','ecommerce','SSM']

def _build_single_turn_prompt(category, concept, price, pack, channels, competitors, market_size, channel_vocab):
    return SINGLE_TURN_PERSONA_PROMPT.format(
        category=category, concept=concept, price=price, pack=pack,
        channels=", ".join(channels), competitors=", ".join(competitors),
        market_size=f"{market_size:,}", channel_vocab=channel_vocab, attribute_list=ATTRIBUTES
    )

def save_single_turn_prompt(category, concept, price, pack, channels, competitors, market_size, channel_vocab):
    text = _build_single_turn_prompt(category, concept, price, pack, channels, competitors, market_size, channel_vocab)
    with open('persona_single_turn_prompt.txt','w',encoding='utf-8') as f: f.write(text)
    if _HAS_RL:
        c = pdfcanvas.Canvas('persona_single_turn_prompt.pdf', pagesize=A4)
        w,h = A4; y=h-50
        for line in text.split('\n'):
            while len(line)>100:
                c.drawString(40,y,line[:100]); y-=14; line=line[100:]; 
                if y<50: c.showPage(); y=h-50
            c.drawString(40,y,line); y-=14
            if y<50: c.showPage(); y=h-50
        c.save()

# ---------- (선택) LLM adapter ----------
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

class LLMAdapter:
    def __init__(self, model="gemini-1.5-flash-latest"):
        if not _HAS_GEMINI:
            raise RuntimeError("google-generativeai 미설치. 무료 사용시 use_llm=False 권장.")
        self.model = model
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 GOOGLE_API_KEY 가 없음.")
        genai.configure(api_key=api_key)
        self.model_client = genai.GenerativeModel(self.model)

    def _extract_json_array(self, text: str) -> list:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"\[\s*{.*}\s*\]", text, flags=re.S)
        if not m:
            raise ValueError("LLM 출력에서 JSON 배열을 찾지 못했습니다.")
        return json.loads(m.group(0))

    def generate(self, n:int, prompt:str):
        msg = prompt.replace("Generate N", f"Generate {n}")
        try:
            rsp = self.model_client.generate_content(msg)
            text = getattr(rsp, "text", None)
            if not text:
                try:
                    parts = rsp.candidates[0].content.parts
                    text = "".join(getattr(p, "text", "") for p in parts)
                except Exception:
                    text = ""
            data = self._extract_json_array(text)
            return [Persona(**d) for d in data]
        except Exception as e:
            print(f"[LLMAdapter] Gemini 파싱 실패: {e} → 규칙기반으로 폴백")
            return None

# ---------- Rule-based personas ----------
GENDERS=['남','여']; REGIONS=['서울','수도권','광역시','기타']
INCOME=['~2천','2-4천','4-7천','7천~']; LIFESTYLES=['활동적','가성비','워라밸','건강지향','육아','미식','트렌디','야근많음']

def _rand_weights()->Dict[str,float]:
    return {a: float(np.clip(np.random.normal(0,0.7),-2,2)) for a in ATTRIBUTES}

def _rand_monthly_pattern()->List[float]:
    base=np.ones(12)
    for k,v in {2:1.08, 8:1.10, 9:1.06}.items(): base[k]*=v  # 설/추석 부근
    base*=np.random.normal(1.0,0.05,12)
    pat=(base/base.mean())
    return list(np.clip(pat,0.6,1.4))

def generate_personas_rule_based(n:int)->List[Persona]:
    out=[]
    for i in range(n):
        out.append(Persona(
            persona_id=f"P{i+1:03d}", name=f"홍길{'동' if i%2==0 else '순'}{i%10}",
            age=int(np.random.randint(20,66)), gender=random.choice(GENDERS),
            region=random.choice(REGIONS), household_size=int(np.random.randint(1,5)),
            income_band=random.choice(INCOME), lifestyle=random.choice(LIFESTYLES),
            health_focus=float(np.clip(np.random.beta(2,3),0,1)),
            price_sensitivity=float(np.clip(np.random.beta(3,2),0,1)),
            brand_loyalty=float(np.clip(np.random.beta(2,4),0,1)),
            online_offline_mix=float(np.random.rand()),
            channel_preference=random.choice(CHANNELS_VOCAB),
            promo_reactivity=float(np.clip(np.random.beta(2,2),0,1)),
            ad_reach_susceptibility=float(np.clip(np.random.beta(2,2),0,1)),
            environmental_concern=float(np.clip(np.random.beta(2,3),0,1)),
            innovation_seeker=float(np.clip(np.random.beta(2,2),0,1)),
            weights=_rand_weights(), monthly_pattern=_rand_monthly_pattern()
        ))
    return out

# ---------- Demand model ----------
@dataclass
class MarketCalendar:
    price_krw:Dict[pd.Period,float]
    discount_rate:Dict[pd.Period,float]
    ad_grps:Dict[pd.Period,float]
    distribution:Dict[pd.Period,float]
    competitor_pressure:Dict[pd.Period,float]
    category_season:Dict[pd.Period,float]

def default_calendar(base_price:int=3500)->MarketCalendar:
    price={m:float(base_price) for m in MONTHS}
    disc={m:(0.12 if m.month in (9,10,2) else 0.0) for m in MONTHS}
    ad={m:200.0 for m in MONTHS}
    for m in MONTHS:
        if m.month in (7,8): ad[m]=400.0
    dist={m:min(1.0,0.35+0.08*i) for i,m in enumerate(MONTHS)}
    comp={m:1.0 for m in MONTHS}
    seas={m:1.0 for m in MONTHS}
    return MarketCalendar(price,disc,ad,dist,comp,seas)

CATEGORY_SEASON = {
    '발효유': [1.05,1.08,1.07,1.03,1.02,1.00,0.98,0.97,0.99,1.02,1.03,1.04],
    '참치':   [0.95,0.98,1.02,1.00,1.03,1.05,1.06,1.08,1.12,1.10,1.03,0.98],
    '조미료': [0.96,0.97,0.98,1.00,1.01,1.02,1.03,1.05,1.08,1.12,1.10,1.00],
    '축산캔': [0.99,1.00,1.01,1.01,1.02,1.02,1.03,1.05,1.06,1.08,1.04,1.00],
    '커피':   [0.98,1.00,1.02,1.03,1.04,1.06,1.08,1.07,1.03,1.00,0.98,0.97],
}
def apply_category_seasonality(cal: MarketCalendar, category2: str):
    key = None
    for k in CATEGORY_SEASON:
        if k in str(category2):
            key = k; break
    if not key: return
    idx = CATEGORY_SEASON[key]
    for i,m in enumerate(MONTHS):
        cal.category_season[m] = float(idx[i])

@dataclass
class SimulationConfig:
    population:int; base_awareness:float; base_trial_rate:float; repeat_rate:float
    price_elasticity:float; ad_effect_per_100grp:float; promo_price_pass_through:float; noise_sd:float=0.05
    product_multiplier: float = 1.0

def _sigmoid(x:float)->float: return 1/(1+math.exp(-x))

def _channel_fit(p: Persona, channels: List[str])->float:
    return 0.1 if p.channel_preference in channels else -0.05

def _feature_match_bonus(p: Persona, feat: str)->float:
    bonus=0.0
    if '고단백' in feat: bonus += 0.15*(p.health_focus - 0.5)
    if ('저당' in feat) or ('저나트륨' in feat): bonus += 0.12*(p.health_focus - 0.5)
    if '락토프리' in feat: bonus += 0.12*(p.health_focus - 0.5)
    if '프리미엄' in feat: bonus += 0.10*(p.innovation_seeker - 0.5)
    return bonus

def _persona_utility(p:Persona, mi:int, cfg:SimulationConfig, cal:MarketCalendar, period:pd.Period,
                     current_channels:List[str], feature_text:str)->float:
    wsum=0.0
    for k,v in p.weights.items():
        val=getattr(p,k,None)
        if val is None: continue
        if isinstance(val,str) and k in ('gender','region','income_band','channel_preference','lifestyle'):
            val_num=(hash((k,val))%7)/6.0
        else:
            val_num=float(val) if not isinstance(val,str) else 0.5
        wsum+=v*val_num
    season=p.monthly_pattern[mi]
    ad=cfg.ad_effect_per_100grp*(cal.ad_grps[period]/100.0)*p.ad_reach_susceptibility
    net_price=cal.price_krw[period]*(1.0-cfg.promo_price_pass_through*cal.discount_rate[period])
    price_term=cfg.price_elasticity*math.log(max(net_price,1.0)/1000.0)*p.price_sensitivity
    comp=math.log(cal.competitor_pressure[period])
    dist=0.1*cal.distribution[period]
    cat_season_term = 0.15*(cal.category_season.get(period,1.0)-1.0)
    channel_term = _channel_fit(p, current_channels)
    feature_term = _feature_match_bonus(p, feature_text)
    return wsum + ad + dist + (-0.3*comp) - 0.5 + 0.2*season + price_term + cat_season_term + channel_term + feature_term

def simulate_monthly_demand(personas:List[Persona], cfg:SimulationConfig, cal:MarketCalendar,
                            current_channels:List[str], feature_text:str,
                            export_csv=True)->pd.DataFrame:
    rows=[]; cum_trials=0.0
    for mi,period in enumerate(MONTHS):
        probs=[]
        for p in personas:
            u=_persona_utility(p,mi,cfg,cal,period,current_channels,feature_text)
            base=_sigmoid(u)
            aware=min(1.0, cfg.base_awareness+0.15*(mi/11.0))
            trial=cfg.base_trial_rate*base
            probs.append(aware*trial*cal.distribution[period])
        exp_trials=cfg.population*np.mean(probs)*cfg.product_multiplier
        cum_trials+=exp_trials
        repeats=cfg.repeat_rate*cum_trials*0.2
        demand=max(0.0,(exp_trials+repeats)*math.exp(np.random.normal(0,cfg.noise_sd)))
        rows.append({'month':period.to_timestamp('M'),
                     'trial_units':exp_trials,'repeat_units':repeats,'total_units':demand,
                     'distribution':cal.distribution[period],'avg_price':cal.price_krw[period],
                     'discount_rate':cal.discount_rate[period],'ad_grps':cal.ad_grps[period],
                     'category_season':cal.category_season.get(period,1.0)})
    df=pd.DataFrame(rows)
    if export_csv: df.to_csv('monthly_forecast.csv',index=False)
    return df

# ---------- 가격 추정 (규칙/모델 통합) ----------
_PREMIUM_TABLE={'프리미엄':0.10,'고단백':0.08,'락토프리':0.07,'저나트륨':0.03,'유기':0.05,'친환경':0.05}
_BASE_UNIT_PRICE={'참치캔':2000/100.0,'액상조미료':900/100.0,'발효유':1100/100.0,'커피-CUP':640/100.0,'고급축산캔':2200/100.0}

def _extract_size(name:str)->Tuple[float,str]:
    m=re.search(r'(\d+(?:\.\d+)?)\s*(g|ml|mL|G|ML)', name)
    if not m:
        if '커피' in name: return 250.0,'mL'
        if '요거트' in name or '발효유' in name: return 400.0,'g'
        if '참치' in name: return 90.0,'g'
        if '조미' in name: return 500.0,'g'
        if '축산' in name: return 200.0,'g'
        return 180.0,'g'
    v,u=float(m.group(1)),m.group(2).lower()
    return v, ('mL' if 'ml' in u else 'g')

def _estimate_list_price_rule(row:pd.Series)->int:
    name=str(row.get('product_name','')); feat=str(row.get('product_feature',''))
    c2=str(row.get('category_level_2','')); c3=str(row.get('category_level_3',''))
    size,unit=_extract_size(name)
    if '참치캔' in c3 or '참치' in c2: per=_BASE_UNIT_PRICE['참치캔']
    elif '조미료' in c3: per=_BASE_UNIT_PRICE['액상조미료']
    elif '발효유' in c2 or '요거트' in name: per=_BASE_UNIT_PRICE['발효유']
    elif '커피' in c2 or 'CUP' in c3: per=_BASE_UNIT_PRICE['커피-CUP']
    elif '축산캔' in c2: per=_BASE_UNIT_PRICE['고급축산캔']
    else: per=1000/100.0
    price=per*size
    prem=sum(v for k,v in _PREMIUM_TABLE.items() if k in feat) + (0.10 if '프리미엄' in name else 0)
    return int(round(price*(1.0+prem),-1))

_PRICE_MODEL_CACHE = None
def _load_price_model():
    global _PRICE_MODEL_CACHE
    if _PRICE_MODEL_CACHE is not None:
        return _PRICE_MODEL_CACHE
    if not (_HAS_JOBLIB and os.path.exists("price_model.pkl")):
        return None
    try:
        _PRICE_MODEL_CACHE = joblib.load("price_model.pkl")
        return _PRICE_MODEL_CACHE
    except Exception as e:
        print("Price model load failed:", e)
        return None

def _estimate_list_price_model(row:pd.Series)->Optional[int]:
    model = _load_price_model()
    if model is None: return None

    name = str(row.get("product_name","")) or ""
    size = int(''.join([c for c in name if c.isdigit()]) or 0)
    unit = 0 if "g" in name.lower() else 1
    feat = str(row.get("product_feature","")) or ""

    X = pd.DataFrame([{
        "pack_size_value": size,
        "pack_size_unit": unit,
        "category_level_1": row.get("category_level_1","NA"),
        "category_level_2": row.get("category_level_2","NA"),
        "category_level_3": row.get("category_level_3","NA"),
        "is_premium": int("프리미엄" in (name+feat) or "premium" in (name+feat).lower()),
        "is_high_protein": int(("고단백" in (name+feat))),
        "is_lactofree": int(("락토프리" in (name+feat)))
    }])

    # dtype 맞추기 (학습과 동일)
    X["pack_size_value"] = pd.to_numeric(X["pack_size_value"], errors="coerce").fillna(0).astype(int)
    X["pack_size_unit"]  = pd.to_numeric(X["pack_size_unit"], errors="coerce").fillna(0).astype(int)
    for c in ["category_level_1","category_level_2","category_level_3"]:
        X[c] = X[c].astype("string").fillna("NA").astype("category")

    try:
        return int(float(model.predict(X)[0]))
    except Exception as e:
        print("Price model predict failed:", e)
        return None

# 최종 통합 진입점: 모델 → 규칙 폴백
def _estimate_list_price(row:pd.Series)->int:
    p = _estimate_list_price_model(row)
    if p is not None: return p
    return _estimate_list_price_rule(row)

# ---------- 텍스트 → 가격/GRP/ACV/유통 강화 ----------
def _parse_month_ranges(text:str)->List[int]:
    res=[]
    for m in re.finditer(r'(\d{1,2})\s*-\s*(\d{1,2})\s*월', text):
        a,b=int(m.group(1)),int(m.group(2)); res+=list(range(a,b+1))
    for m in re.finditer(r'(?<!-)\b(\d{1,2})\s*월', text):
        res.append(int(m.group(1)))
    return sorted(set([x for x in res if 1<=x<=12]))

def enrich_calendar_from_features(cal:MarketCalendar, feat:str, name:str)->None:
    months={m.month:m for m in MONTHS}; lower=feat.lower()
    for m in MONTHS: cal.ad_grps[m]=200.0
    if '광고 x' in lower or '광고x' in lower:
        for m in MONTHS: cal.ad_grps[m]=120.0
    if any(k in lower for k in ['광고 진행','tv','youtube','sns']):
        for mm in _parse_month_ranges(feat):
            if mm in months: cal.ad_grps[months[mm]]=max(cal.ad_grps[months[mm]],450.0)
    if '엘리베이터 광고' in feat:
        for mm in _parse_month_ranges(feat):
            if mm in months: cal.ad_grps[months[mm]]+=100.0
    if 'sns 바이럴' in feat:
        for mm in _parse_month_ranges(feat):
            if mm in months: cal.ad_grps[months[mm]]=max(cal.ad_grps[months[mm]],250.0)
    for m in MONTHS: cal.discount_rate[m]=0.12 if m.month in (9,10,2) else 0.0
    if any(k in feat for k in ['행사','프로모션','기획']):
        for mm in _parse_month_ranges(feat):
            if mm in months: cal.discount_rate[months[mm]]=min(0.2, cal.discount_rate[months[mm]]+0.05)
    start,step=0.35,0.08
    if any(k in name for k in ['CUP','컵','커피']): start,step=0.45,0.10
    if '엘리베이터 광고' in feat and any(mm in (6,7,8) for mm in _parse_month_ranges(feat)): start+=0.05
    cal.distribution={m:min(1.0,start+step*i) for i,m in enumerate(MONTHS)}

# ---------- Heuristics for channels/competitors/markets ----------
def _infer_channels(c1,c2,c3):
    if '발효유' in str(c2): return ['hypermarket','convenience']
    if '참치' in str(c2):   return ['hypermarket','SSM','ecommerce']
    if '조미료' in str(c3): return ['hypermarket','ecommerce']
    if '축산캔' in str(c2): return ['hypermarket','SSM','ecommerce']
    if '커피' in str(c2):   return ['convenience','hypermarket']
    return ['hypermarket','ecommerce']

def _infer_competitors(c2):
    if '참치' in str(c2): return ['CJ','오뚜기','사조']
    if '조미'  in str(c2): return ['CJ','오뚜기']
    if '발효유' in str(c2): return ['빙그레','매일','남양']
    if '축산캔' in str(c2): return ['SPAM','롯데','동원']
    if '커피' in str(c2):   return ['매일','동서','스타벅스RTD']
    return ['CJ','오뚜기','사조']

def _infer_market_size(name:str,c2:str='')->int:
    base=6_000_000
    if '발효유' in c2 or '요거트' in name: base=3_000_000
    elif '참치' in c2: base=10_000_000
    elif '조미' in c2: base=5_000_000
    elif '축산캔' in c2: base=4_000_000
    elif '커피' in c2: base=12_000_000
    h=(abs(hash(name))%41)/100.0
    return int(base*(0.8+h))

# ---------- Calibrated params ----------
CALIB_PATH = "calibrated_theta.json"
def load_calibrated_theta()->Optional[dict]:
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ---------- Persona cache utilities (NEW) ----------
def _persona_to_dict(p: Persona) -> dict:
    return {
        "persona_id": p.persona_id, "name": p.name,
        "age": p.age, "gender": p.gender, "income_band": p.income_band,
        "region": p.region, "household_size": p.household_size, "lifestyle": p.lifestyle,
        "health_focus": p.health_focus, "price_sensitivity": p.price_sensitivity,
        "brand_loyalty": p.brand_loyalty, "online_offline_mix": p.online_offline_mix,
        "channel_preference": p.channel_preference, "promo_reactivity": p.promo_reactivity,
        "ad_reach_susceptibility": p.ad_reach_susceptibility,
        "environmental_concern": p.environmental_concern, "innovation_seeker": p.innovation_seeker,
        "weights": p.weights, "monthly_pattern": p.monthly_pattern,
    }

def _personas_from_dict_list(lst: list[dict]) -> list[Persona]:
    return [Persona(**d) for d in lst]

def _scenario_signature(category, concept, price, pack, channels, competitors, market_size, persona_count, model_name="gemini-1.5-flash-latest"):
    sig = {
        "category": category, "concept": concept, "price": price, "pack": pack,
        "channels": list(channels), "competitors": list(competitors),
        "market_size": int(market_size), "persona_count": int(persona_count),
        "model": model_name,
    }
    s = json.dumps(sig, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def _persona_cache_path(cache_dir:str, cache_key:str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"personas_{cache_key}.json")

def save_personas_cache(personas: list[Persona], cache_dir: str, cache_key: str):
    path = _persona_cache_path(cache_dir, cache_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([_persona_to_dict(p) for p in personas], f, ensure_ascii=False, indent=2)
    print(f"[persona-cache] saved -> {path}")

def load_personas_cache(cache_dir: str, cache_key: str) -> Optional[list[Persona]]:
    path = _persona_cache_path(cache_dir, cache_key)
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    personas = _personas_from_dict_list(data)
    print(f"[persona-cache] loaded <- {path} ({len(personas)} personas)")
    return personas

# ---------- Scenario runner ----------
def run_scenario(persona_count:int=400, use_llm:bool=False,
                 category:str='식품', concept:str='신제품',
                 price:int=3500, pack:str='unit',
                 channels:List[str]=['hypermarket','convenience','ecommerce'],
                 competitors:List[str]=['CJ','오뚜기','사조'],
                 market_size:int=3_200_000,
                 calendar:Optional[MarketCalendar]=None,
                 ad_effect_mult:float=1.0,
                 feature_text:str="",
                 persona_cache_dir:str="persona_cache",
                 force_llm_regen:bool=False
                 ):
    save_single_turn_prompt(category, concept, price, pack, channels, competitors, market_size, CHANNELS_VOCAB)

    personas = None

    # 캐시 키 계산 및 로드
    cache_key = _scenario_signature(
        category, concept, price, pack, channels, competitors, market_size,
        persona_count, model_name="gemini-1.5-flash-latest"
    )
    if not force_llm_regen:
        personas = load_personas_cache(persona_cache_dir, cache_key)

    # LLM 생성 + 캐시 저장
    if personas is None:
        if use_llm:
            try:
                adapter = LLMAdapter()
                personas = adapter.generate(persona_count,
                    _build_single_turn_prompt(category, concept, price, pack, channels, competitors, market_size, CHANNELS_VOCAB)
                )
                if personas:
                    save_personas_cache(personas, persona_cache_dir, cache_key)
            except Exception as e:
                print(f"[run_scenario] LLM 사용 실패: {e} → 규칙기반으로 진행")
                personas = None

    # 규칙기반 폴백
    if personas is None:
        personas = generate_personas_rule_based(persona_count)

    cal = calendar or default_calendar(price)

    theta = load_calibrated_theta()
    if theta:
        base_awareness    = float(theta.get("base_awareness", 0.18))
        base_trial_rate   = float(theta.get("base_trial_rate", 0.35))
        repeat_rate       = float(theta.get("repeat_rate", 0.55))
        price_elasticity  = float(theta.get("price_elasticity", -1.2))
        ad_effect_k       = float(theta.get("ad_effect_per_100grp", 0.10))
        promo_pass        = float(theta.get("promo_price_pass_through", 0.8))
        product_mult      = float(theta.get("product_multiplier", 1.0))
    else:
        base_awareness, base_trial_rate, repeat_rate = 0.18, 0.35, 0.55
        price_elasticity, ad_effect_k, promo_pass = -1.2, 0.10, 0.8
        product_mult = 1.0

    cfg = SimulationConfig(
        population=market_size,
        base_awareness=base_awareness,
        base_trial_rate=base_trial_rate,
        repeat_rate=repeat_rate,
        price_elasticity=price_elasticity,
        ad_effect_per_100grp=ad_effect_k,
        promo_price_pass_through=promo_pass,
        noise_sd=0.06,
        product_multiplier=product_mult
    )
    forecast=simulate_monthly_demand(personas, cfg, cal, channels, feature_text, export_csv=True)
    SOURCES.to_json('sources_used.json'); _write_solution_outline(forecast, persona_count, category, concept, price, pack)
    return forecast, personas

def _write_solution_outline(forecast:pd.DataFrame, persona_count:int, category:str, concept:str, price:int, pack:str):
    last=forecast.tail(1).iloc[0]
    md=(f"# 솔루션 설명 자료(초안)\n"
        f"## 개요\n- 카테고리:{category}\n- 컨셉:{concept}\n- 가격:{price:,} / {pack}\n- 페르소나:{persona_count}\n"
        f"## 핵심결과\n- 12M 합계:{int(forecast['total_units'].sum()):,} EA\n"
        f"- 런칭월:{int(forecast.iloc[0]['total_units']):,} EA\n- 최종월:{int(last['total_units']):,} EA\n")
    with open('solution_report.md','w',encoding='utf-8') as f: f.write(md)

# ---------- One product → 12m (MC 앙상블) ----------
def forecast_one_product(row:pd.Series, persona_count:int=400, use_llm:bool=False, mc_runs:int=7)->np.ndarray:
    global RNG
    name=str(row.get('product_name','')).strip()
    feat=str(row.get('product_feature','')).strip()
    c1=str(row.get('category_level_1','')); c2=str(row.get('category_level_2','')); c3=str(row.get('category_level_3',''))

    seed=abs(hash(name))%(2**32-1)
    price=_estimate_list_price(row)
    market_size=_infer_market_size(name,c2); channels=_infer_channels(c1,c2,c3); competitors=_infer_competitors(c2)
    cal=default_calendar(price); enrich_calendar_from_features(cal, feat, name); apply_category_seasonality(cal, c2)

    ad_mult=1.0
    if '광고모델' in feat and any(k in feat for k in ['안유진','아이돌','연예인']): ad_mult=1.15
    size,unit=_extract_size(name); pack=f"{int(size)}{unit}"

    preds=[]
    for r in range(mc_runs):
        RNG=np.random.default_rng(seed + r*1337); random.seed(seed + r*7331)
        df,_=run_scenario(persona_count, use_llm, c1 or '식품', feat[:80] or '신제품',
                          price, pack, channels, competitors, market_size, cal, ad_mult, feature_text=feat)
        preds.append(np.maximum(0, np.array(df['total_units'].values)))
    y = np.rint(np.median(np.stack(preds, axis=0), axis=0)).astype(int)
    return y

# ---------- Submission ----------
def make_submission(product_info_csv:str, out_csv:str='submission.csv', persona_count:int=400,
                    use_llm:bool=False, mc_runs:int=7)->pd.DataFrame:
    prod=pd.read_csv(product_info_csv)
    rows=[]
    for _,row in prod.iterrows():
        y=forecast_one_product(row, persona_count, use_llm, mc_runs=mc_runs)
        rows.append({'product_name':row['product_name'],
                     **{f'months_since_launch_{i+1}':int(y[i]) for i in range(12)}})
    sub=pd.DataFrame(rows)
    sub.to_csv(out_csv,index=False)
    return sub

# ---------- (선택) 간단 캘리브레이션 ----------
def simple_calibrate(y_true: np.ndarray,
                     personas: List[Persona],
                     cal_template: MarketCalendar,
                     init: dict = None,
                     bounds: dict = None):
    from scipy.optimize import minimize
    init = init or dict(base_awareness=0.18, base_trial_rate=0.35, repeat_rate=0.55,
                        price_elasticity=-1.2, ad_effect_per_100grp=0.10,
                        promo_price_pass_through=0.8, product_multiplier=1.0)
    bounds = bounds or dict(
        base_awareness=(0.05,0.6), base_trial_rate=(0.05,0.8), repeat_rate=(0.2,0.8),
        price_elasticity=(-2.5,-0.2), ad_effect_per_100grp=(0.02,0.25),
        promo_price_pass_through=(0.5,0.95), product_multiplier=(0.6,1.6)
    )

    keys = list(init.keys())
    x0 = np.array([init[k] for k in keys])
    lo = np.array([bounds[k][0] for k in keys])
    hi = np.array([bounds[k][1] for k in keys])

    channels = ['hypermarket','convenience','ecommerce']
    feat_text = ""

    def pack(theta):
        return {k: float(v) for k,v in zip(keys, theta)}

    def clamp(theta):
        return np.minimum(np.maximum(theta, lo), hi)

    def simulate_theta(theta):
        th = pack(theta)
        cfg = SimulationConfig(
            population=3_000_000,
            base_awareness=th['base_awareness'],
            base_trial_rate=th['base_trial_rate'],
            repeat_rate=th['repeat_rate'],
            price_elasticity=th['price_elasticity'],
            ad_effect_per_100grp=th['ad_effect_per_100grp'],
            promo_price_pass_through=th['promo_price_pass_through'],
            noise_sd=0.0,
            product_multiplier=th['product_multiplier']
        )
        df = simulate_monthly_demand(personas, cfg, cal_template, channels, feat_text, export_csv=False)
        return df['total_units'].values

    def loss(theta):
        theta = clamp(theta)
        y_pred = simulate_theta(theta)
        mape = np.mean(np.abs(y_true - y_pred) / (y_true + 1))
        reg = 1e-3*np.sum((theta - x0)**2)
        return float(mape + reg)

    res = minimize(loss, x0, method='Nelder-Mead', options={'maxiter':300})
    theta_opt = pack(clamp(res.x))
    with open(CALIB_PATH,'w',encoding='utf-8') as f: json.dump(theta_opt,f,ensure_ascii=False,indent=2)
    print("Saved calibrated params ->", CALIB_PATH)
    return theta_opt

# ===================== 가격 수집 & 모델 학습 유틸 =====================

# Step1. category_prices.csv 생성 (네이버 쇼핑 API 예시)
def naver_price_search(query, client_id=None, client_secret=None):
    import requests
    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": client_id or os.environ.get("NAVER_CLIENT_ID",""),
        "X-Naver-Client-Secret": client_secret or os.environ.get("NAVER_CLIENT_SECRET",""),
    }
    params = {"query": query, "display": 3, "sort": "sim"}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200: return None
    items = resp.json().get("items", [])
    if not items: return None
    prices = []
    for it in items:
        try:
            prices.append(int(it.get("lprice")))
        except Exception:
            pass
    return min(prices) if prices else None

def build_category_prices(product_info_csv, out_csv="category_prices.csv",
                          client_id=None, client_secret=None):
    prod = pd.read_csv(product_info_csv)
    rows = []
    today = dt.date.today().isoformat()
    for _, row in prod.iterrows():
        name = str(row["product_name"])
        price = None
        try:
            price = naver_price_search(name, client_id, client_secret)
        except Exception as e:
            print("naver_price_search error:", e)
        rows.append({
            "product_name": name,
            "category_level_1": row.get("category_level_1",""),
            "category_level_2": row.get("category_level_2",""),
            "category_level_3": row.get("category_level_3",""),
            "pack_size_value": int(''.join([c for c in name if c.isdigit()]) or 0),
            "pack_size_unit": "g" if "g" in name.lower() else "ml",
            "list_price": price,
            "channel": "네이버쇼핑",
            "observed_at": today
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)
    return df

# Step2. LightGBM 학습
def train_price_model(csv_path="category_prices.csv"):
    if not _HAS_LGBM or not _HAS_JOBLIB:
        raise RuntimeError("lightgbm/joblib 미설치. `pip install lightgbm joblib` 후 재시도하세요.")
    df = pd.read_csv(csv_path).dropna(subset=["list_price"])
    if df.empty:
        raise ValueError("학습 가능한 행이 없습니다(list_price NaN).")

    y = df["list_price"].astype(float).values

    # ▶︎ feature 준비
    X = df[["pack_size_value","pack_size_unit","category_level_1",
            "category_level_2","category_level_3","product_name"]].copy()

    # 수치형
    X["pack_size_value"] = pd.to_numeric(X["pack_size_value"], errors="coerce").fillna(0).astype(int)
    X["pack_size_unit"] = X["pack_size_unit"].map({"g":0,"ml":1,"mL":1}).fillna(0).astype(int)

    # 키워드 파생
    pn = X["product_name"].astype(str).fillna("")
    X["is_premium"]     = pn.str.contains("프리미엄", na=False).astype(int)
    X["is_high_protein"]= pn.str.contains("고단백", na=False).astype(int)
    X["is_lactofree"]   = pn.str.contains("락토프리", na=False).astype(int)

    # 문자열 카테고리 → pandas category
    for c in ["category_level_1","category_level_2","category_level_3"]:
        X[c] = X[c].astype("string").fillna("NA").astype("category")

    # 더 이상 product_name은 사용 안 함
    X = X.drop(columns=["product_name"])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ▶︎ LightGBM에 카테고리 컬럼 알려주기
    cat_cols = ["category_level_1","category_level_2","category_level_3"]

    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7)
    model.fit(X_train, y_train, categorical_feature=cat_cols)

    print("Train R2:", model.score(X_train, y_train))
    print("Test  R2:", model.score(X_test, y_test))

    joblib.dump(model, "price_model.pkl")
    print("Saved price_model.pkl")
    return model

# ===================== CLI =====================
if __name__ == '__main__':
    import sys
    # Jupyter/Colab/노트북이면 CLI 파싱을 건너뜀
    if 'ipykernel' in sys.modules or 'JPY_PARENT_PID' in os.environ:
        print("[forecast_pipeline] Loaded in Jupyter. Use the Python functions directly (no CLI).")
    else:
        import argparse
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd")

        p_gp = sub.add_parser("gather_prices")
        p_gp.add_argument("--product_csv", required=True)
        p_gp.add_argument("--out_csv", default="category_prices.csv")
        p_gp.add_argument("--naver_client_id", default=os.environ.get("NAVER_CLIENT_ID"))
        p_gp.add_argument("--naver_client_secret", default=os.environ.get("NAVER_CLIENT_SECRET"))

        p_tm = sub.add_parser("train_price_model")
        p_tm.add_argument("--prices_csv", default="category_prices.csv")

        p_ms = sub.add_parser("make_submission")
        p_ms.add_argument("--product_csv", required=True)
        p_ms.add_argument("--out_csv", default="submission.csv")
        p_ms.add_argument("--persona_count", type=int, default=400)
        p_ms.add_argument("--use_llm", type=int, default=0)   # 0/1
        p_ms.add_argument("--mc_runs", type=int, default=7)
        p_ms.add_argument("--force_llm_regen", type=int, default=0)
        p_ms.add_argument("--persona_cache_dir", default="persona_cache")

        args = p.parse_args()

        if args.cmd == "gather_prices":
            build_category_prices(args.product_csv, args.out_csv, args.naver_client_id, args.naver_client_secret)

        elif args.cmd == "train_price_model":
            train_price_model(args.prices_csv)

        elif args.cmd == "make_submission":
            subdf = make_submission(args.product_csv, args.out_csv,
                                    persona_count=args.persona_count,
                                    use_llm=bool(args.use_llm),
                                    mc_runs=args.mc_runs)
            print(subdf.head())
            print("Files: submission.csv, monthly_forecast.csv, persona_single_turn_prompt.txt/.pdf, sources_used.json, solution_report.md")

        else:
            # 기본 동작
            try:
                subdf = make_submission('product_info.csv','submission.csv',
                                        persona_count=400, use_llm=False, mc_runs=7)
                print(subdf.head())
                print("Files: submission.csv, monthly_forecast.csv, persona_single_turn_prompt.txt/.pdf, sources_used.json, solution_report.md")
            except Exception as e:
                print("Run with a subcommand. Error:", e)
