# BTC实盘交易系统

基于OKX API和自定义策略的BTC实时交易系统，支持多种技术分析策略，自动执行交易并记录详细信息。

## 功能特性

- 🔄 **多策略支持**: 集成趋势跟踪、RSI反转、布林带等多种交易策略
- 📊 **实时交易**: 基于OKX API进行BTC-USDT永续合约交易
- 📈 **技术指标**: 支持EMA、ATR、RSI、布林带等技术指标
- 💰 **风险管理**: 内置止损止盈、仓位管理、杠杆控制
- 📝 **详细记录**: 自动记录每笔交易的详细信息
- 📋 **交易报告**: 生成每日/每周/每月交易总结报告
- 🔧 **灵活配置**: 支持自定义交易参数和策略选择

## 支持的交易策略

### 1. trend_atr_signal - 趋势跟踪策略
- **原理**: 基于EMA金叉死叉和ATR动态止盈止损
- **开仓**: EMA短期上穿长期（金叉）做多，反之做空
- **平仓**: 基于ATR计算的动态止盈止损价格
- **适用**: 趋势性较强的市场

### 2. boll_rsi_signal - 布林带RSI策略
- **原理**: 结合布林带位置和RSI超买超卖信号
- **开仓**: 价格触及布林带下轨且RSI超卖时做多，反之做空
- **平仓**: 价格回归布林带中轨或RSI回归正常区间
- **适用**: 震荡市场中的反转交易

### 3. rsi_reversal_signal - RSI反转策略
- **原理**: 纯RSI超买超卖反转信号
- **开仓**: RSI从超卖区域反弹时做多，从超买区域回落时做空
- **平仓**: RSI达到相反的极值区域
- **适用**: 明显的超买超卖市场

### 4. trend_volatility_stop_signal - 趋势波动止损策略
- **原理**: 趋势跟踪结合基于ATR的波动性止损
- **开仓**: EMA金叉死叉信号
- **平仓**: 基于ATR的移动止损
- **适用**: 高波动性市场

## 安装和配置

### 1. 环境要求
```bash
python >= 3.8
pandas
pandas-ta
okx-python
loguru
```

### 2. 安装依赖
```bash
pip install pandas pandas-ta okx-python loguru
```

### 3. 配置API密钥
创建配置文件 `trading_config.json`:
```json
{
    "api_key": "your_okx_api_key",
    "secret_key": "your_okx_secret_key",
    "passphrase": "your_okx_passphrase",
    "symbol": "BTC-USDT-SWAP",
    "strategy": "trend_atr",
    "trade_mode": "cross",
    "position_size": 0.001,
    "max_positions": 1,
    "leverage": 5,
    "timeframe": "5m",
    "data_limit": 100,
    "risk_management": {
        "max_loss_per_trade": 0.02,
        "max_daily_loss": 0.05,
        "stop_loss_atr_multiplier": 1.5,
        "take_profit_atr_multiplier": 2.0
    }
}
```

### 4. 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| api_key | OKX API密钥 | 必填 |
| secret_key | OKX Secret密钥 | 必填 |
| passphrase | OKX 密码短语 | 必填 |
| symbol | 交易对 | BTC-USDT-SWAP |
| strategy | 交易策略 | trend_atr |
| trade_mode | 交易模式 | cross |
| position_size | 仓位大小(BTC) | 0.001 |
| leverage | 杠杆倍数 | 5 |
| timeframe | K线周期 | 5m |
| data_limit | 获取数据条数 | 100 |

## 使用方法

### 1. 快速开始
```bash
# 运行演示（了解系统功能）
python trading_example.py

# 配置API密钥后开始实盘交易
python btc_live_trader.py
```

### 2. 演示模式
运行 `trading_example.py` 可以查看：
- 各种策略的回测效果
- 交易器初始化流程
- 报告生成示例

### 3. 实盘交易
配置好API密钥后，可以直接运行实盘交易：
```bash
python btc_live_trader.py
```

程序会：
1. 自动获取最新市场数据
2. 根据选择的策略生成交易信号
3. 自动执行买入/卖出操作
4. 记录每笔交易详情
5. 生成交易报告

### 4. 连续交易模式
程序支持连续运行，定期检查交易信号：
```python
from btc_live_trader import BTCLiveTrader

trader = BTCLiveTrader("trading_config.json")
trader.start_continuous_trading(interval_seconds=300)  # 每5分钟检查一次
```

## 交易记录和报告

### 1. 交易记录
系统会自动记录以下信息：
- 交易时间戳和ID
- 交易方向（买入/卖出）
- 交易价格和数量
- 策略名称和信号类型
- 手续费和盈亏
- 账户余额变化

### 2. 报告文件
- `trade_records_YYYYMMDD.json`: 详细交易记录
- `balance_history_YYYYMMDD.csv`: 余额变化历史
- `daily_report_YYYYMMDD.json`: 每日交易报告

### 3. 报告内容
每日报告包含：
- 交易统计（总次数、胜率、盈亏比）
- 当前持仓状态
- 未实现盈亏
- 详细交易列表
- 策略表现分析

## 风险管理

### 1. 仓位管理
- 默认每次交易使用固定仓位大小
- 支持设置最大持仓数量
- 可配置杠杆倍数

### 2. 止损止盈
- ATR动态止损
- 可配置止损/止盈倍数
- 支持移动止损策略

### 3. 风险控制
- 单笔最大亏损限制
- 每日最大亏损限制
- 紧急平仓机制

## 策略参数调优

### 1. trend_atr策略参数
```python
# 在strategy.py中调整参数
signals = trend_atr_signal(
    df,
    short_ema=8,      # 短期EMA周期
    long_ema=21,      # 长期EMA周期
    atr_len=14,       # ATR计算周期
    tp_atr=2.0,       # 止盈ATR倍数
    sl_atr=1.0        # 止损ATR倍数
)
```

### 2. boll_rsi策略参数
```python
signals = boll_rsi_signal(
    df,
    bb_len=20,        # 布林带周期
    bb_std=2.0,       # 布林带标准差
    rsi_len=14        # RSI周期
)
```

## 注意事项

### ⚠️ 风险提示
1. **高风险警告**: 数字货币交易存在极高风险，可能导致全部资金损失
2. **谨慎投资**: 请只用闲置资金进行交易，确保能够承受损失
3. **充分测试**: 建议先在测试环境或使用小额资金测试策略
4. **市场风险**: 策略表现受市场环境影响，过往表现不代表未来收益
5. **技术风险**: API故障、网络问题可能影响交易执行

### 📝 使用建议
1. **小额开始**: 建议从小仓位开始测试
2. **分散投资**: 不要将所有资金投入单一策略
3. **定期监控**: 定期检查交易表现和账户状态
4. **及时止损**: 设置合理的止损机制，控制亏损
5. **持续学习**: 了解市场动态，优化交易策略

### 🛡️ 安全提醒
1. **API安全**: 妥善保管API密钥，不要泄露给他人
2. **权限控制**: 建议为交易API设置适当的权限限制
3. **IP白名单**: 在OKX平台设置IP白名单
4. **定期更换**: 定期更换API密钥
5. **监控异常**: 监控账户异常活动

## 故障排除

### 常见问题

**Q: API连接失败怎么办？**
A: 检查API密钥配置，确认网络连接正常，检查OKX服务状态

**Q: 交易信号生成但未执行？**
A: 检查账户余额是否充足，确认仓位设置合理，查看交易模式配置

**Q: 策略表现不佳？**
A: 考虑调整策略参数，选择更适合当前市场的策略，降低仓位大小

**Q: 如何停止连续交易？**
A: 按 Ctrl+C 可以安全停止程序，系统会自动生成最终报告

### 日志查看
系统会生成详细的日志文件 `btc_trading.log`，包含：
- 交易执行详情
- 错误和警告信息
- 策略信号记录
- 系统运行状态

## 更新日志

### v1.0.0
- 初始版本发布
- 支持4种交易策略
- 完整的交易记录和报告功能
- 风险管理系统

## 技术支持

如有问题或建议，请：
1. 查看日志文件获取详细错误信息
2. 检查API配置和网络连接
3. 参考OKX官方API文档
4. 确认策略参数设置合理

---
1. *_equity.csv（权益曲线）                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                    
  - datetime: 时间戳（K线时间）。                                                                                                                                                                                                                                                                                   
  - equity: 当时总权益（现金 + 持仓市值/空头开仓资金 + 未实现盈亏；未扣已记录的开仓手续费）。                                                                                                                                                                                                                       
  - position: 持仓方向（0 无仓，1 多头，-1 空头）。                                                                                                                                                                                                                                                                 
  - price: 当根K线的收盘价。                                                                                                                                                                                                                                                                                        
  - daily_return: 本行权益相对上一行的收益率（equity.pct_change()），首行为空。                                                                                                                                                                                                                                     
  - cummax: 到当前为止权益的历史最高值。                                                                                                                                                                                                                                                                            
  - drawdown: 回撤比率 = (equity - cummax) / cummax（负数表示回撤）。                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                    
  2. *_trades.csv（成交记录）                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                    
  - datetime: 该笔成交的时间。                                                                                                                                                                                                                                                                                      
  - action: 成交类型（如 BUY/SELL/SELL_SHORT/BUY_TO_COVER）。                                                                                                                                                                                                                                                       
  - price: 成交价（已含滑点）。                                                                                                                                                                                                                                                                                     
  - shares: 成交数量。                                                                                                                                                                                                                                                                                              
  - type: 头寸方向标签（Long/Short）。                                                                                                                                                                                                                                                                              
  - commission: 该笔交易产生的手续费。                                                                                                                                                                                                                                                                              
  - pnl: 该笔平仓的已实现盈亏（仅出现在平仓行，开仓行为空）。  