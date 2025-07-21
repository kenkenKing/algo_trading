# https://quantpedia.com/strategies/market-timing-with-aggregate-accruals/
#
# The value-weighted S&P 500 Index and yields on the 3-month T-Bills are used as proxies for aggregate stock market returns and risk-free rates. 
# Accruals for each company in the S&P 500 index are calculated each year at the end of April as:
# BS_ACC = ( ∆CA – ∆Cash) – ( ∆CL – ∆STD – ∆ITP) – Dep
# Where:
# ∆CA = annual change in current assets
# ∆Cash = change in cash and cash equivalents
# ∆CL = change in current liabilities
# ∆STD = change in debt included in current liabilities
# ∆ITP = change in income taxes payable
# Dep = annual depreciation and amortization expense
# The value-weighted aggregate accruals are then calculated. The term premium is also calculated as the yield spread of a ten-year Treasury Bond
# over a one-month Treasury Bill. Aggregate accruals and the term spread are then used in a regression equation (with 15 years of data) to calculate 
# the next year’s forecast for the mean and variance of excess stock market returns. The weight of investment in the stock market is proportional to
# the forecasted mean-variance ratio of stock returns, and the remaining cash is invested in T-Bills. The portfolio is held for twelve months, and
# it is rebalanced at the end of April (yearly rebalancing frequency).
#
# QC implementation changes:
#   - Aggregate accruals are calculated using 500 most liquid US stocks' accruals.

#region imports
from AlgorithmImports import *
from collections import deque
import numpy as np
from scipy import stats
from math import sqrt
#endregion

class MarketTimingAggregateAccruals(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        self.sp500_stocks:List = []
        
        # monthly market prices
        self.market_monthly_data = deque(maxlen = 12)

        # latest accruals data
        self.accrual_data:Dict[Symbol, float] = {}
        
        self.coarse_count:int = 500
        self.year_period:int = 10
        self.gamma:float = 5.
        self.leverage:int = 5
        self.leverage_cap:float = 3.
        
        self.aggregate_accruals = deque(maxlen = self.year_period + 1)
        
        # yearly market data -> price and variance pair
        self.market_yearly_data = deque(maxlen = self.year_period + 1)
        
        self.market:Symbol = self.AddEquity('SPY', Resolution.Daily).Symbol
        self.tbills:Symbol = self.AddEquity('BIL', Resolution.Daily).Symbol
        
        self.selection_flag:bool = False
        self.UniverseSettings.Resolution = Resolution.Daily
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0
        self.AddUniverse(self.FundamentalSelectionFunction)
        self.Schedule.On(self.DateRules.MonthStart(self.market), self.TimeRules.AfterMarketOpen(self.market), self.Selection)

        self.settings.daily_precise_end_time = False

        self.set_warm_up(timedelta(20 * 12 * 31))
    def Selection(self) -> None:
        # store market price
        if self.Securities.ContainsKey(self.market) and self.Securities[self.market]:
            self.market_monthly_data.append(self.Securities[self.market].Price)
            
        if self.Time.month == 4:
        # if self.Time.month in [4,10]:
            self.selection_flag = True

            if len(self.market_monthly_data) == self.market_monthly_data.maxlen:
                # store yearly market data
                yearly_volatility:float = (self.Volatility([x for x in self.market_monthly_data]) * sqrt(len(self.market_monthly_data)))
                variance:float = yearly_volatility ** 2
                self.market_yearly_data.append((self.Securities[self.market].Price, variance))
    
    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            security.SetFeeModel(CustomFeeModel())
            security.SetLeverage(self.leverage)

        for security in changes.RemovedSecurities:
            if security.Symbol in self.accrual_data:
                del self.accrual_data[security.Symbol]

    def FundamentalSelectionFunction(self, fundamental: List[Fundamental]) -> List[Symbol]:
        if not self.selection_flag:
            return Universe.Unchanged

        # selected = [x.Symbol for x in coarse if x.HasFundamentalData and x.Market == 'usa']
        selected = [x for x in sorted([x for x in fundamental if x.HasFundamentalData and x.Market == 'usa' \
                           and (x.FinancialStatements.BalanceSheet.CurrentAssets.HasValue and \
                                x.FinancialStatements.BalanceSheet.CashAndCashEquivalents.HasValue and \
                                x.FinancialStatements.BalanceSheet.CurrentLiabilities.HasValue and \
                                x.FinancialStatements.BalanceSheet.CurrentDebt.HasValue and \
                                x.FinancialStatements.BalanceSheet.IncomeTaxPayable.HasValue and \
                                x.FinancialStatements.IncomeStatement.DepreciationAndAmortization.HasValue)],
                key = lambda x: x.DollarVolume, reverse = True)[:self.coarse_count]]

        accruals_market_cap:dict[Symbol, Tuple] = {}
        for stock in selected:
            symbol:Symbol = stock.Symbol
            if symbol not in self.accrual_data:
                self.accrual_data[symbol] = None
                
            # accrual calculation
            current_accruals_data:AccrualsData = AccrualsData(stock.FinancialStatements.BalanceSheet.CurrentAssets.Value, stock.FinancialStatements.BalanceSheet.CashAndCashEquivalents.Value,
                                                            stock.FinancialStatements.BalanceSheet.CurrentLiabilities.Value, stock.FinancialStatements.BalanceSheet.CurrentDebt.Value, stock.FinancialStatements.BalanceSheet.IncomeTaxPayable.Value,
                                                            stock.FinancialStatements.IncomeStatement.DepreciationAndAmortization.Value, stock.FinancialStatements.BalanceSheet.TotalAssets.Value)
            
            # there is not previous accrual data
            if not self.accrual_data[symbol]:
                self.accrual_data[symbol] = current_accruals_data
                continue
            
            # accruals and market cap calculation
            accruals:float = self.CalculateAccruals(current_accruals_data, self.accrual_data[symbol])
            market_cap:float = stock.MarketCap
            accruals_market_cap[symbol] = (accruals, market_cap)
    
            # update accruals data
            self.accrual_data[symbol] = current_accruals_data

        if len(accruals_market_cap) == 0: return Universe.Unchanged
        
        # value weighted accruals calculation
        total_market_cap:float = sum([x[1][1] for x in accruals_market_cap.items()])
        weighted_accruals_data:List[float] = []
        for symbol, accruals_and_cap in accruals_market_cap.items():
            weight:float = accruals_and_cap[1] / total_market_cap
            weighted_accruals_data.append(accruals_and_cap[0] * weight)
            
        aggregate_accruals:float = sum([x for x in weighted_accruals_data])
        self.aggregate_accruals.append(aggregate_accruals)

        return list(self.accrual_data.keys())

    def OnData(self, data:Slice) -> None:
        if not self.selection_flag:
            return
        self.selection_flag = False

        # 10 years of accruals history is ready
        if len(self.market_yearly_data) == self.market_yearly_data.maxlen and \
            len(self.aggregate_accruals) == self.aggregate_accruals.maxlen:

            # Regression calc.
            market_prices:np.ndarray = np.array([x[0] for x in self.market_yearly_data])
            market_returns:np.ndarray = (market_prices[1:] - market_prices[:-1]) / market_prices[:-1]
            
            # shift values for regression to predict future return
            market_returns:np.ndarray = market_returns[-(self.year_period - 1):]
            accruals:List[float] = [x for x in self.aggregate_accruals]
            regr_accruals = accruals[:(self.year_period - 1)]
            
            # Simple Linear Regression
            # Y = α + (β ∗ X)
            slope, intercept, r_value, p_value, std_err = stats.linregress(regr_accruals, market_returns)
            expected_return:float = intercept + slope * accruals[-1]

            # predict variance
            yearly_variances:List[float] = [x[1] for x in self.market_yearly_data]
            yearly_variances = yearly_variances[-(self.year_period - 1):]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(regr_accruals, yearly_variances)
            expected_variance:float = intercept + slope * accruals[-1]
            
            market_weight:float = expected_return / (expected_variance * self.gamma)
            market_weight = max(min(market_weight, self.leverage_cap), -self.leverage_cap) # leverage cap
            tbills_weight:float = 1. - abs(market_weight) if abs(market_weight) < 1. else 0.
                    
            # trade execution
            if self.market in data and data[self.market] and self.tbills in data and data[self.tbills]:
                self.SetHoldings(self.market, market_weight)
                self.SetHoldings(self.tbills, tbills_weight)
    
    def Volatility(self, values) -> float:
        values:np.ndarray = np.array(values)
        returns:np.ndarray = (values[1:] - values[:-1]) / values[:-1]
        return np.std(returns)
    
    def CalculateAccruals(self, current_accrual_data, prev_accrual_data) -> float:
        delta_assets:float = current_accrual_data.CurrentAssets - prev_accrual_data.CurrentAssets
        delta_cash:float = current_accrual_data.CashAndCashEquivalents - prev_accrual_data.CashAndCashEquivalents
        delta_liabilities:float = current_accrual_data.CurrentLiabilities - prev_accrual_data.CurrentLiabilities
        delta_debt:float = current_accrual_data.CurrentDebt - prev_accrual_data.CurrentDebt
        delta_tax:float = current_accrual_data.IncomeTaxPayable - prev_accrual_data.IncomeTaxPayable
        dep:float = current_accrual_data.DepreciationAndAmortization
        avg_total:float = (current_accrual_data.TotalAssets + prev_accrual_data.TotalAssets) / 2
        
        bs_acc:float = ((delta_assets - delta_cash) - (delta_liabilities - delta_debt - delta_tax) - dep)
        return bs_acc

# custom fee model
class CustomFeeModel(FeeModel):
    def GetOrderFee(self, parameters):
        fee:float = parameters.Security.Price * parameters.Order.AbsoluteQuantity * 0.00005
        return OrderFee(CashAmount(fee, "USD"))

class AccrualsData():
    def __init__(self, current_assets:float, cash_and_cash_equivalents:float, current_liabilities:float, current_debt:float, income_tax_payable:float, depreciation_and_amortization:float, total_assets:float):
        self.CurrentAssets:float = current_assets
        self.CashAndCashEquivalents:float = cash_and_cash_equivalents
        self.CurrentLiabilities:float = current_liabilities
        self.CurrentDebt:float = current_debt
        self.IncomeTaxPayable:float = income_tax_payable
        self.DepreciationAndAmortization:float = depreciation_and_amortization
        self.TotalAssets:float = total_assets