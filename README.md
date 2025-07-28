Sample of code files from one of my current projects: an investment strategy backtester.

- Recommended to start by reviewing backtest.py
- Several test are included
- This public project will not run due to missing structure, input data, and dependencies

I'm happy to provide a live demo or additional code samples if desired


Full project filetree:

<pre lang="markdown">
project_root/
├── raw_data/
│   ├── rfr/
│       └── rfr.csv
├── resources/
│   ├── sys_config_files/
│       └── etl_system_settings.json
├── src/
│   ├── core/
│   │   ├── Dashboard/
│   │   │   └── Dashboard.py
│   │   ├── backtesting/
│   │   │   ├── analyzer_metrics.py
│   │   │   ├── backtest.py
│   │   │   ├── backtest_analyzer.py
│   │   │   ├── backtest_analyzer_v2.py
│   │   │   ├── order_factory.py
│   │   │   ├── order_handler.py
│   │   │   └── trade_cost_model.py
│   │   ├── utilities/
│   │   │   ├── mass_file_operations.py
│   │   │   ├── time_utils.py
│   │   │   └── utils.py
│   │   ├── config_parsers.py
│   │   ├── environment_initialization.py
│   │   └── logging_config.py
│   ├── extractors/
│   │   ├── dremio_extractor.py
│   │   ├── file_extractors.py
│   │   └── filestore_extractor.py
│   ├── tests/
│   │   ├── input/
│   │   │   ├── test_backtest_history.csv
│   │   │   ├── test_prices.csv
│   │   │   └── test_rfr.csv
│   │   ├── create_synthetic_test_data.py
│   │   ├── test_analyzer_metrics.py
│   │   ├── test_backtest.py
│   │   ├── test_backtest_analyzer.py
│   │   ├── test_backtest_analyzer_v2.py
│   │   ├── test_fee_calculator.py
│   │   └── test_trade_cost_model.py
│   ├── visualizations/
│       └── viz.py
├── README.md
└── requirements.txt
 </pre>