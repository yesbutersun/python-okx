# OKX沙盒环境交易指南

本指南详细介绍如何使用OKX沙盒环境进行安全的BTC交易测试。

## 🔒 什么是沙盒环境

沙盒环境（Sandbox）是OKX提供的测试交易环境，具有以下特点：

- **测试资金**: 使用虚拟资金，无真实资金风险
- **完整功能**: 所有API接口与生产环境完全一致
- **真实市场**: 连接真实的市场数据和价格
- **安全测试**: 可以安全测试策略和交易逻辑
- **无手续费**: 通常不收取真实交易手续费

## 🌐 沙盒环境配置

### 环境参数
```
API域名: https://www.okx.com
环境标记: flag = "0" (沙盒)
交易标记: sandbox = true
```

### 与生产环境对比
| 参数 | 沙盒环境 | 生产环境 |
|------|----------|----------|
| API域名 | 相同 | 相同 |
| flag参数 | "0" | "1" |
| 资金 | 虚拟资金 | 真实资金 |
| 风险 | 无资金风险 | 资金风险 |
| 手续费 | 通常免除 | 实际收取 |

## 🚀 快速开始

### 方法1: 使用专用配置脚本
```bash
python setup_sandbox_trading.py
```

该脚本会：
- 指导获取沙盒API密钥
- 提供多种测试策略选择
- 自动生成安全的配置参数
- 测试API连接

### 方法2: 使用主配置界面
```bash
python start_btc_trading.py
```
选择"2. 沙盒环境"选项

### 方法3: 手动配置文件
创建 `sandbox_trading_config.json`:
```json
{
    "api_key": "your_sandbox_api_key",
    "secret_key": "your_sandbox_secret_key",
    "passphrase": "your_sandbox_passphrase",
    "domain": "https://www.okx.com",
    "flag": "0",
    "environment": "sandbox",
    "symbol": "BTC-USDT-SWAP",
    "strategy": "rsi_reversal",
    "trade_mode": "cross",
    "position_size": 0.0001,
    "leverage": 2,
    "timeframe": "5m",
    "data_limit": 100
}
```

## 📋 获取沙盒API密钥

### 步骤详解
1. **访问OKX官网**
   - 打开 [www.okx.com](https://www.okx.com)
   - 登录您的账户

2. **创建API**
   - 进入"API管理"页面
   - 点击"创建API Key"
   - **重要**: 选择沙盒环境(Sandbox)

3. **配置权限**
   - 交易权限: ✅ 启用
   - 读取权限: ✅ 启用
   - 提现权限: ❌ 禁用（沙盒不需要）

4. **安全设置**
   - 设置IP白名单（推荐）
   - 记录API Key、Secret Key、Passphrase

### 重要提醒
- 沙盒API密钥与生产环境完全独立
- 需要单独申请沙盒环境的API
- 测试资金由系统自动提供

## 💰 沙盒资金管理

### 初始资金
- 通常提供 **100,000 USDT** 测试资金
- 支持所有主流交易对
- 资金可以随时重置

### 杠杆设置
- 支持 1-125倍杠杆
- 建议测试时使用较低杠杆（1-5倍）
- 可以测试不同杠杆倍数的效果

### 余额查看
```python
from okx import Account

# 初始化沙盒账户API
account_api = Account.AccountAPI(
    api_key=api_key,
    api_secret_key=secret_key,
    passphrase=passphrase,
    domain="https://www.okx.com"
)

balance = account_api.get_balance()
print(f"账户余额: {balance}")
```

## 🎯 推荐测试策略

### 1. 保守测试策略
- **策略**: RSI反转
- **仓位**: 0.0001 BTC
- **杠杆**: 2倍
- **适合**: 初次使用，验证基本功能

### 2. 趋势测试策略
- **策略**: 趋势ATR
- **仓位**: 0.0005 BTC
- **杠杆**: 5倍
- **适合**: 测试趋势跟踪效果

### 3. 均衡测试策略
- **策略**: 布林RSI
- **仓位**: 0.001 BTC
- **杠杆**: 3倍
- **适合**: 综合测试，平衡风险和收益

## 🧪 测试流程建议

### 第一阶段: 基础测试
1. **API连接测试**
   ```python
   python -c "
   from btc_live_trader import BTCLiveTrader
   trader = BTCLiveTrader('sandbox_trading_config.json')
   print(f'环境: {trader.environment}')
   price = trader.get_current_price()
   print(f'当前价格: ${price}')
   "
   ```

2. **策略信号测试**
   ```bash
   python validate_strategies.py
   ```

3. **单次交易测试**
   ```python
   from btc_live_trader import BTCLiveTrader
   trader = BTCLiveTrader('sandbox_trading_config.json')
   trader.run_trading_cycle()  # 运行一次交易周期
   ```

### 第二阶段: 连续交易测试
```bash
# 小间隔测试（每30秒检查一次）
python btc_live_trader.py --config sandbox_trading_config.json --interval 30

# 正常间隔测试（每5分钟检查一次）
python btc_live_trader.py --config sandbox_trading_config.json --interval 300
```

### 第三阶段: 策略优化
- 监控交易表现
- 调整策略参数
- 测试不同市场环境
- 验证止盈止损功能

## 📊 监控和分析

### 实时监控
```bash
# 查看实时日志
tail -f btc_trading.log

# 查看交易记录
tail -f trade_records_$(date +%Y%m%d).json
```

### 性能分析
```bash
# 生成每日报告
python -c "
from btc_live_trader import BTCLiveTrader
trader = BTCLiveTrader('sandbox_trading_config.json')
report = trader.generate_daily_report()
print(report)
"
```

### 策略回测
```bash
# 运行策略回测验证
python validate_strategies.py
```

## ⚠️ 注意事项

### 安全提醒
1. **API密钥安全**
   - 沙盒API密钥也要妥善保管
   - 不要在代码中硬编码密钥
   - 定期更换API密钥

2. **环境区分**
   - 明确标识沙盒配置文件
   - 避免与生产环境配置混淆
   - 测试时确认环境标记

### 功能限制
1. **资金限制**
   - 虚拟资金，不能提取
   - 部分高级功能可能受限
   - 数据可能定期重置

2. **市场延迟**
   - 市场数据与真实环境可能略有延迟
   - 高频交易测试要考虑延迟影响
   - 滑点可能与真实环境不同

### 最佳实践
1. **从小开始**
   - 使用极小仓位测试
   - 逐步增加仓位大小
   - 验证每笔交易的准确性

2. **完整测试**
   - 测试开仓、平仓功能
   - 验证止盈止损逻辑
   - 检查异常处理机制

3. **记录分析**
   - 记录所有测试交易
   - 分析策略表现指标
   - 总结优化方向

## 🔄 从沙盒到生产环境

### 迁移步骤
1. **策略验证**
   - 在沙盒环境充分测试
   - 确认策略稳定盈利
   - 验证风险控制有效

2. **生产环境准备**
   - 申请生产环境API密钥
   - 准备真实资金
   - 设置风险限制

3. **配置更新**
   ```bash
   python start_btc_trading.py
   # 选择"1. 生产环境"
   # 使用已验证的策略参数
   ```

4. **谨慎开始**
   - 使用比沙盒更小的仓位
   - 密切监控初期表现
   - 准备应急平仓计划

## 🆘 故障排除

### 常见问题

**Q: API连接失败**
```
解决方案:
1. 检查网络连接
2. 验证API密钥正确性
3. 确认选择了沙盒环境
4. 检查IP白名单设置
```

**Q: 余额显示为0**
```
解决方案:
1. 等待系统自动分配测试资金
2. 检查是否正确选择沙盒环境
3. 尝试重新初始化API客户端
4. 联系OKX技术支持
```

**Q: 策略不产生交易信号**
```
解决方案:
1. 检查市场数据获取是否正常
2. 验证策略参数设置合理
3. 降低信号触发阈值
4. 增加数据获取量
```

**Q: 交易执行失败**
```
解决方案:
1. 检查账户余额充足
2. 验证交易参数正确
3. 确认交易对在沙盒环境可用
4. 检查杠杆设置是否超限
```

### 技术支持
- **OKX官方文档**: [https://www.okx.com/docs-v5/](https://www.okx.com/docs-v5/)
- **沙盒环境支持**: 通过OKX官方客服渠道
- **开发者社区**: [https://www.okx.com/community](https://www.okx.com/community)

## 📚 相关资源

### 项目文件
- `setup_sandbox_trading.py` - 沙盒环境专用配置脚本
- `btc_live_trader.py` - 主要交易逻辑
- `validate_strategies.py` - 策略验证工具
- `trading_example.py` - 使用示例
- `start_btc_trading.py` - 用户友好界面

### 配置示例
- `sandbox_trading_config.json` - 沙盒环境配置模板
- `trading_config.json` - 通用配置文件
- `strategy_validation_report.json` - 策略验证报告

### 日志文件
- `btc_trading.log` - 主要交易日志
- `trade_records_YYYYMMDD.json` - 每日交易记录
- `balance_history_YYYYMMDD.csv` - 余额变化历史

---

## 总结

沙盒环境是验证交易策略和测试交易逻辑的理想场所。通过在沙盒环境的充分测试，您可以：

✅ **安全验证** - 无资金风险的策略测试
✅ **功能测试** - 验证所有交易功能
✅ **参数优化** - 找到最佳策略参数
✅ **风险控制** - 测试风险管理机制
✅ **经验积累** - 熟悉交易操作流程

建议在沙盒环境进行至少1-2周的充分测试，确认策略稳定有效后再考虑进入生产环境。

**记住**: 在金融交易中，保护资金永远是第一位的！