# LLM 기반 자동화 코인 트레이드 시스템 (Bithumb)

이 프로젝트는 Bithumb 거래소를 대상으로 한 **LLM 기반 의사결정 자동매매 엔진**을 제공합니다. 
공개 시세/호가 데이터와 계좌 정보를 수집하고, 언어모델이 생성한 트레이딩 결정을 리스크 매니저가 검증하여 
주문 집행(또는 드라이런)까지 자동으로 수행할 수 있는 구조를 갖추고 있습니다.

## 주요 구성 요소

| 모듈 | 역할 |
| --- | --- |
| `bithumb_llm_trader.api_client` | Bithumb 공개/개인 REST API 래퍼 및 시그니처 생성 |
| `bithumb_llm_trader.config` | 전략/위험/LLM 설정 로더 및 데이터 클래스 |
| `bithumb_llm_trader.prompts` | 시장/계좌 정보를 기반으로 프롬프트 생성 |
| `bithumb_llm_trader.decision` | LLM 출력 JSON 파싱 및 검증 로직 |
| `bithumb_llm_trader.risk` | 리스크 한도(신뢰도·포지션·손절/익절) 적용 |
| `bithumb_llm_trader.engine` | 전체 파이프라인(데이터 수집 → LLM 의사결정 → 리스크 검증 → 주문) 조율 |
| `bithumb_llm_trader.multi_agent` | 복수 전략/자산을 아우르는 멀티 에이전트 포트폴리오 매니저 |
| `bithumb_llm_trader.main` | CLI 실행 진입점 (`python -m bithumb_llm_trader.main`) |

모듈은 모두 독립적으로 테스트 가능하도록 설계되었으며, `tests/` 디렉터리의 Pytest 스위트가 핵심 동작을 검증합니다.

## 설정 파일 작성

전략 설정은 JSON/YAML/TOML 파일로 정의할 수 있습니다. 예시(`config.sample.json`):

```json
{
  "api": {
    "api_key": "YOUR_BITHUMB_API_KEY",
    "api_secret": "YOUR_BITHUMB_API_SECRET",
    "base_url": "https://api.bithumb.com",
    "timeout": 10.0
  },
  "trading_pair": {
    "order_currency": "BTC",
    "payment_currency": "KRW"
  },
  "risk": {
    "max_trade_value": 1000000,
    "max_position_size": 0.2,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.03,
    "min_confidence": 0.6
  },
  "llm": {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_output_tokens": 512
  },
  "dry_run": true
}
```

* `dry_run` 값을 `true`로 두면 주문을 실행하지 않고 시뮬레이션만 수행합니다.
* YAML(`.yml`, `.yaml`) 또는 TOML(`.toml`) 포맷도 동일한 필드 구조를 따릅니다.

## 실행 방법

1. (선택) OpenAI API 키를 환경 변수에 설정합니다.
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
2. 설정 파일을 준비합니다. (위 JSON 예시 활용)
3. 아래 명령으로 단일 사이클을 실행할 수 있습니다.
   ```bash
   python -m bithumb_llm_trader.main path/to/config.json --verbose
   ```
   * `--openai-api-key` 옵션으로 환경 변수 대신 키를 직접 전달할 수 있습니다.
   * 실행 결과는 JSON 형식으로 표준 출력에 표시되며, `dry_run`이 `false`이면 실제 주문이 전송됩니다.

## 동작 흐름

1. **데이터 수집** – `BithumbAPI`가 시세/호가/잔고를 조회합니다.
2. **프롬프트 생성** – `build_trading_prompt`가 시장·계좌·리스크 정보를 정리합니다.
3. **LLM 의사결정** – `LLMDecisionMaker`가 언어모델 응답을 받아 `TradeDecision`으로 파싱합니다.
4. **리스크 필터링** – `RiskManager`가 신뢰도·자금·포지션 한도를 점검하고 손절/익절 가격을 부여합니다.
5. **주문 실행** – 드라이런 여부에 따라 실제 주문을 전송하거나, 실행 정보만 기록합니다.
6. **이력 관리** – 최근 의사결정/실행 내역은 `TradingEngine.history`에 보관됩니다.

### 포트폴리오 레벨 멀티 에이전트 오케스트레이션

`MultiAgentPortfolioManager`는 여러 `StrategyConfig` 묶음을 동시에 실행할 수 있는 상위 레이어입니다. 전략마다 LLM 결정을 내리고(`LLMDecisionMaker`), 리스크 매니저로 보정한 뒤(`RiskManager`), 실행 에이전트가 실제 주문 또는 드라이런을 수행합니다. 각 전략의 결과(`StrategyCycleResult`)와 현금/포지션 집계가 `PortfolioCycleResult`로 반환되어 포트폴리오 차원의 의사결정 및 리밸런싱에 활용할 수 있습니다.

## 테스트

Pytest 스위트가 주요 로직(시그니처 생성, 의사결정 파서, 리스크 매니저, 엔진 플로우)을 검증합니다.

```bash
pytest
```

## 주의 사항

- 실제 운용 시 **API 키 보안**과 **LLM 응답 검증**을 반드시 강화해야 합니다.
- 언어모델 호출에는 비용이 발생하며, 응답 지연을 고려한 추가 설계(비동기, 큐 관리 등)가 필요할 수 있습니다.
- 본 예제 코드는 참고용으로 제공되며, 실거래 전 충분한 시뮬레이션과 검증이 선행되어야 합니다.
