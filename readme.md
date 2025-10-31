# Code Portfolio

This repository showcases two Python projects demonstrating production-ready software engineering across financial systems and data infrastructure.

## Projects

### Data Build Tool (`dbt/`)
An ETL framework with intelligent API throttling and concurrent data operations.

**Technical Highlights:**
- **Concurrency**: Multi-threaded extraction with thread-safe rate limiting
- **API Management**: Configurable throttling system (per-second/minute/day limits)
- **Pipeline Architecture**: Modular ETL components with pluggable data sources
- **Production Features**: Centralized logging, secrets management, error handling

**Start here:** `src/io/throttler.py`, `src/io/data_extractor.py`

---

### Investment Strategy Backtester (`back_tester/`)
A backtesting engine simulating trading strategies with realistic order execution and cost modeling.

**Technical Highlights:**
- **OOP & Design Patterns**: Factory pattern for order types, ABC for polymorphic behavior, strategy pattern for order lifecycle management
- **Financial Modeling**: Multiple order types (Market, Limit, Stop, Stop-Limit), execution simulation, comprehensive fee calculations
- **State Management**: Order lifecycle tracking with expiration and trigger conditions
- **Testing**: Unit tests with synthetic data generation

**Start here:** `src/backtest.py`

---

## Technical Skills Demonstrated
- Clean architecture with separation of concerns
- Comprehensive testing and error handling
- Type hints and Python best practices
- Configuration-driven design
- Version control and documentation

---

**Note:** Code samples are sanitized for public sharing. Dependencies and data sources removed. Happy to discuss implementation details or provide a live demo of dbt (including full ETLs).