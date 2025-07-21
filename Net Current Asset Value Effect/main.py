# https://quantpedia.com/strategies/net-current-asset-value-effect/
#
# The investment universe consists of all stocks on the London Exchange. Companies with more than one class of ordinary shares and foreign companies 
# are excluded. Also excluded are companies on the lightly regulated markets and companies which belong to the financial sector. The portfolio of
# stocks is formed annually in July. Only those stocks with an NCAV/MV higher than 1.5 are included in the NCAV/MV portfolio. This Buy-and-hold
# portfolio is held for one year. Stocks are weighted equally.
#
# QC implementation changes:
#   - Instead of all listed London stocks, we selected top 3000 US listed stocks by market cap from QC stock universe.

from AlgorithmImports import *
import numpy as np

class NetCurrentAssetValueEffect(QCAlgorithm):

    def Initialize(self) -> None:
        self.SetStartDate(2000, 1, 1)  
        self.SetCash(100_000) 

        self.UniverseSettings.Leverage = 3
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.FundamentalFunction)
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0.0
        self.settings.daily_precise_end_time = False

        # Fundamental Filter Parameters
        self.fundamental_count: int = 3_000
        self.market: str = 'usa'
        self.country_id: str = 'USA'
        self.fin_sector_code: int = 103
        self.ncav_threshold: float = 1.5
        
        self.long_symbols: List[Symbol] = []

        self.rebalance_month: int = 7   
        self.selection_flag: bool = True

        self.exchange: Symbol = self.AddEquity('SPY', Resolution.Daily).Symbol
        self.Schedule.On(self.DateRules.MonthStart(self.exchange), 
                        self.TimeRules.AfterMarketOpen(self.exchange), 
                        self.Selection)

    def FundamentalFunction(self, fundamental: List[Fundamental]) -> List[Symbol]:
        if not self.selection_flag:
            return Universe.Unchanged

        filtered: List[Fundamental] = [f for f in fundamental if f.HasFundamentalData
                                        and f.Market == self.market
                                        and f.CompanyReference.CountryId == self.country_id
                                        and f.AssetClassification.MorningstarSectorCode != self.fin_sector_code
                                        and not np.isnan(f.EarningReports.BasicAverageShares.TwelveMonths)
                                        and f.EarningReports.BasicAverageShares.TwelveMonths != 0
                                        and not np.isnan(f.MarketCap)
                                        and f.MarketCap != 0
                                        and not np.isnan(f.ValuationRatios.WorkingCapitalPerShare)
                                        and f.ValuationRatios.WorkingCapitalPerShare != 0
                                        ]

        sorted_by_market_cap: List[Fundamental] = sorted(filtered, 
                                                        key=lambda f: f.MarketCap, 
                                                        reverse=True)[:self.fundamental_count]

        # Calculate NCAV/MV
        self.long_symbols = [x.Symbol for x in sorted_by_market_cap 
                            if ((x.ValuationRatios.WorkingCapitalPerShare * 
                            x.EarningReports.BasicAverageShares.TwelveMonths) / x.MarketCap) > self.ncav_threshold]

        return self.long_symbols
    
    def OnData(self, slice: Slice) -> None:
        if not self.selection_flag:
            return
        self.selection_flag = False

        # Trade Execution
        portfolio: List[PortfolioTarget] = [PortfolioTarget(symbol, 1 / len(self.long_symbols)) 
                                            for symbol in self.long_symbols 
                                            if slice.ContainsKey(symbol) and slice[symbol] is not None]

        self.SetHoldings(portfolio, True)
        self.long_symbols.clear()
    
    def Selection(self) -> None:
        if self.Time.month == self.rebalance_month:
            self.selection_flag = True

    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        for security in changes.AddedSecurities:
            security.SetFeeModel(CustomFeeModel())

# Custom fee model
class CustomFeeModel(FeeModel):
    def GetOrderFee(self, parameters: OrderFeeParameters) -> OrderFee:
        fee: float = parameters.Security.Price * parameters.Order.AbsoluteQuantity * 0.00005
        return OrderFee(CashAmount(fee, "USD"))