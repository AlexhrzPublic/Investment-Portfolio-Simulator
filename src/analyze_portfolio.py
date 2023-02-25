import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import math
from dash import html
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt


class PortfolioAnalyzer:
    def __init__(self, positions, start_date):
        """
        Initializes the PortfolioAnalyzer class

        Args:
            positions (dict): A dictionary of positions in the portfolio where the key is a string, the ticker,
                and the value is a float, the dollar amount of the position.
            start_date (datetime.datetime): The date to start the analyses from.
        """
        self.positions = positions
        self.start_date = start_date
        self.close, self.open_, self.low, self.high, self.shares, self.index, self.tickers = self._process_data()

    def graph_portfolio(self) -> go.Figure:
        """
        Graphs the portfolio using plotly

        Returns:
            go.Figure: A plotly figure where the x-axis is time and the y-axis is the dollar value of the portfolio.

        """

        # Get the portfolio value
        portfolio_close = np.sum([self.close[ticker] * self.shares[ticker] for ticker in self.tickers], axis=0)
        portfolio_open = np.sum([self.open_[ticker] * self.shares[ticker] for ticker in self.tickers], axis=0)
        portfolio_high = np.sum([self.high[ticker] * self.shares[ticker] for ticker in self.tickers], axis=0)
        portfolio_low = np.sum([self.low[ticker] * self.shares[ticker] for ticker in self.tickers], axis=0)

        # Create the figure
        fig = go.Figure()

        # Add the portfolio trace with the date on the x-axis and the dollar value on the y-axis
        fig.add_trace(go.Scatter(
            x=self.index,
            y=portfolio_close,
            mode="lines",
            name="Portfolio"))

        # Add candlestick graph for the entire portfolio
        fig.add_trace(go.Candlestick(x=self.index,
                                     open=portfolio_open,
                                     high=portfolio_high,
                                     low=portfolio_low,
                                     close=portfolio_close,
                                     visible=False,
                                     name="Close",
                                     showlegend=False))

        fig = PortfolioAnalyzer._configure_layout(fig)
        fig.update_layout(title_text='Portfolio Details', title_y=1, title_x=0.7)

        # Return the figure
        return fig

    def graph_individual_stocks(self) -> go.Figure:
        """
        Graphs the portfolio using plotly

        Returns:
            go.Figure: A plotly figure where the x-axis is time and the y-axis is the dollar value of the portfolio.

        """

        # Create the figure
        fig = go.Figure()

        # Add the position traces with the date on the x-axis and the dollar value on the y-axis
        for ticker in self.tickers:
            fig.add_trace(go.Scatter(
                x=self.index,
                y=self.close[ticker] * self.shares[ticker],
                name=ticker))

            # Graph the candlestick chart for each stock in the portfolio
            fig.add_trace(go.Candlestick(x=self.index,
                                         open=self.open_[ticker],
                                         high=self.high[ticker],
                                         low=self.low[ticker],
                                         close=self.close[ticker],
                                         visible=False,
                                         name="Close",
                                         showlegend=False))

        fig = PortfolioAnalyzer._configure_layout(fig)
        fig.update_layout(title_text='Individual Details', title_y=1, title_x=0.7)

        # Return the figure
        return fig

    def exponential_smoothing(self, predict_year) -> go.Figure:
        """
        Graphs the smoothing exponential prediction using plotly
        
        Args: 
            predict_year(int): The starting year of the prediction
        Returns:
            go.Figure: A plotly figure where the x-axis is time and the y-axis is the dollar value of the exponential prediction.

        """
        lst = []
        stock = pd.Series(lst)
        for ticker in self.tickers:
            # Get the data for the tickers for the last 5 years
            data = yf.download(ticker,
                               start=self.start_date,
                               end=datetime.datetime.now(),
                               adjusted=True,
                               progress=False)

            # summarize to monthly frequency
            stock_ = data.resample('M').last().rename(columns={'Adj Close': 'adj_close'}).adj_close
            # combine each stock as portfolio data
            stock = pd.concat([stock, stock_], axis=0).sum(level=0).sort_index(inplace=False)

        # create a training/test set
        train_indices = stock.index.year < predict_year
        stock_train = stock[train_indices]
        stock_test = stock[~train_indices]
        test_length = len(stock_test)

        # fit three simple exponential smoothing (SES) models
        # and create predictions for them
        # smoothing level 0.2
        ses_1 = SimpleExpSmoothing(stock_train).fit(smoothing_level=0.2)
        ses_forecast_1 = ses_1.forecast(test_length)
        s1 = pd.concat([ses_1.fittedvalues, ses_forecast_1], axis=0)

        # smoothing level 0.5
        ses_2 = SimpleExpSmoothing(stock_train).fit(smoothing_level=0.5)
        ses_forecast_2 = ses_2.forecast(test_length)
        s2 = pd.concat([ses_2.fittedvalues, ses_forecast_2], axis=0)

        # the optimal smoothing level selected by the statsmodels' optimizer
        ses_3 = SimpleExpSmoothing(stock_train).fit()
        alpha = ses_3.model.params['smoothing_level']
        ses_forecast_3 = ses_3.forecast(test_length)
        s3 = pd.concat([ses_3.fittedvalues, ses_forecast_3], axis=0)

        # visual the original price and model
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock.index, y=stock.values,
                                 line=dict(color='black'),
                                 mode='lines',
                                 name='Actual'))

        fig.add_trace(go.Scatter(x=s1.index, y=s1.values,
                                 line=dict(color='firebrick'),
                                 mode='lines',
                                 name=r'$\alpha=0.2$'))

        fig.add_trace(go.Scatter(x=s2.index, y=s2.values,
                                 line=dict(color='royalblue'),
                                 mode='lines',
                                 name=r'$\alpha=0.5$'))

        fig.add_trace(go.Scatter(x=s3.index, y=s3.values,
                                 line=dict(color='darkseagreen'),
                                 mode='lines',
                                 name=r'$\alpha={0:.4f}$'.format(alpha)))

        fig.update_layout(title='Portfolio Simple Exponential Smoothing',
                          xaxis_title='Date',
                          yaxis_title='Value',
                          title_y=1, title_x=0.7)
        return fig

    def holt_smoothing(self, predict_year) -> go.Figure:
        """
        Graphs the holt smoothing prediction using plotly
        
        Args: 
            predict_year(int): The starting year of the prediction
        Returns:
            go.Figure: A plotly figure where the x-axis is time and the y-axis is the dollar value of the holt prediction.

        """
        lst = []
        stock = pd.Series(lst)
        for ticker in self.tickers:
            # Get the data for the tickers for the last 5 years
            data = yf.download(ticker,
                               start=self.start_date,
                               end=datetime.datetime.now(),
                               adjusted=True,
                               progress=False)

            # summarize to monthly frequency
            stock_ = data.resample('M').last().rename(columns={'Adj Close': 'adj_close'}).adj_close
            # combine each stock as portfolio data
            stock = pd.concat([stock, stock_], axis=0).sum(level=0).sort_index(inplace=False)

        # create a training/test set
        train_indices = stock.index.year < predict_year
        stock_train = stock[train_indices]
        stock_test = stock[~train_indices]
        test_length = len(stock_test)

        # Create three Holt smoothing models and predictions
        # Holt's model with linear trend
        hs_1 = Holt(stock_train).fit()
        hs_forecast_1 = hs_1.forecast(test_length)
        h1 = pd.concat([hs_1.fittedvalues, hs_forecast_1], axis=0)

        # Holt's model with exponential trend
        hs_2 = Holt(stock_train, exponential=True).fit()
        hs_forecast_2 = hs_2.forecast(test_length)
        h2 = pd.concat([hs_2.fittedvalues, hs_forecast_2], axis=0)

        # Holt's model with exponential trend and damping
        hs_3 = Holt(stock_train, exponential=False,
                    damped=True).fit(damping_trend=0.99)
        hs_forecast_3 = hs_3.forecast(test_length)
        h3 = pd.concat([hs_3.fittedvalues, hs_forecast_3], axis=0)

        # visual the original price and model
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=stock.index, y=stock.values,
                                 line=dict(color='black'),
                                 mode='lines',
                                 name='Actual'))

        fig.add_trace(go.Scatter(x=h1.index, y=h1.values,
                                 line=dict(color='firebrick'),
                                 mode='lines',
                                 name='Linear trend'))

        fig.add_trace(go.Scatter(x=h2.index, y=h2.values,
                                 line=dict(color='royalblue'),
                                 mode='lines',
                                 name='Exponential trend'))

        fig.add_trace(go.Scatter(x=h3.index, y=h3.values,
                                 line=dict(color='darkseagreen'),
                                 mode='lines',
                                 name='Exponential trend (damped)'))

        fig.update_layout(title=ticker + " Holt's Smoothing models",
                          xaxis_title='Date',
                          yaxis_title='Value',
                          title_y=1, title_x=0.7)
        return fig

    def get_portfolio_stats(self) -> html.Div:
        """
        Gets the statistics for the portfolio since the start date, including the start value, the end value,
        the total return, the percent change, volatility, and Sharpe ratio

        Returns:
            html.Div: A Div containing the statistics for the portfolio

        """

        # Get the shares of each position
        shares = {ticker: self.positions[ticker] / self.close[ticker][0] for ticker in self.tickers}

        # Get the portfolio value by multiplying the close price by the number of shares
        portfolio = pd.DataFrame(np.sum([self.close[ticker] * shares[ticker] for ticker in self.tickers], axis=0))
        portfolio.index = self.close.index

        # Get the start value
        start_value = portfolio.iloc[0][0]

        # Get the end value
        end_value = portfolio.iloc[-1][0]

        # Get the total return
        total_return = end_value - start_value

        # Get the percent change
        percent_change = total_return / start_value

        # Get the volatility
        volatility = float(portfolio.pct_change().std())

        # Get the Sharpe ratio
        sharpe_ratio = float(portfolio.pct_change().mean(axis=0) / volatility)
        sharpe_ratio *= np.sqrt(252)

        # Calculate the Sharpe ratio interpretation
        if sharpe_ratio > 3:
            sharpe_ratio_interpretation = "★★★★★"
        elif sharpe_ratio > 2:
            sharpe_ratio_interpretation = "★★★★☆"
        elif sharpe_ratio > 1:
            sharpe_ratio_interpretation = "★★★☆☆"
        elif sharpe_ratio > 0:
            sharpe_ratio_interpretation = "★★☆☆☆"
        else:
            sharpe_ratio_interpretation = "★☆☆☆☆"

        # Get the best and worst days (both the value and the date)
        best_day_value = portfolio.max()[0]
        best_day_date = portfolio.idxmax()[0]
        best_day_date = best_day_date.strftime("%Y-%m-%d")
        worst_day_value = portfolio.min()[0]
        worst_day_date = portfolio.idxmin()[0]
        worst_day_date = worst_day_date.strftime("%Y-%m-%d")

        # Create the Div
        div = html.Div([
            html.H3("Portfolio Statistics"),
            html.Table([
                html.Tr([
                    html.Td("Start Value", className="input-table-column-right-padding"),
                    html.Td(f"${start_value:,.2f}")
                ]),
                html.Tr([
                    html.Td("End Value", className="input-table-column-right-padding"),
                    html.Td(f"${end_value:,.2f}")
                ]),
                html.Tr([
                    html.Td("Total Return", className="input-table-column-right-padding"),
                    html.Td(f"${total_return:,.2f}")
                ]),
                html.Tr([
                    html.Td("Percent Change", className="input-table-column-right-padding"),
                    html.Td(f"{percent_change:.2%}")
                ]),
                html.Tr([
                    html.Td("Best Day", className="input-table-column-right-padding"),
                    html.Td(f"${best_day_value:,.2f} on {best_day_date}")
                ]),
                html.Tr([
                    html.Td("Worst Day", className="input-table-column-right-padding"),
                    html.Td(f"${worst_day_value:,.2f} on {worst_day_date}")
                ]),
                html.Tr([
                    html.Td("Volatility", className="input-table-column-right-padding"),
                    html.Td(f"{volatility:.2%}")
                ]),
                html.Tr([
                    html.Td("Sharpe Ratio", className="input-table-column-right-padding"),
                    html.Td(f"{sharpe_ratio:.2f} — {sharpe_ratio_interpretation}")
                ])
            ], className="stats-table")
        ], style={"margin": "30px 0"})

        # Return the Div
        return div

    def get_stock_stats(self) -> html.Div:
        """
        Gets the statistics for each stock since the start date, including the start value, the end value,
        the total return, and the percent change.

        Returns:
            html.Div: A table containing the statistics for each stock.

        """
        # Create the Div
        div = html.Div([
            html.H3("Stock Statistics"),
            html.Table([
                           html.Tr([
                               html.Td("Ticker", className="stock-table-column-right-padding"),
                               html.Td("Start Value", className="stock-table-column-right-padding"),
                               html.Td("End Value", className="stock-table-column-right-padding"),
                               html.Td("Total Return", className="stock-table-column-right-padding"),
                               html.Td("Percent Change", className="stock-table-column-right-padding")
                           ])
                       ] + [
                           html.Tr([
                               html.Td(ticker, className="stock-table-column-right-padding"),
                               html.Td(f"${self.positions[ticker]:,.2f}"),
                               html.Td(f"${self.close[ticker][-1] * self.shares[ticker]:,.2f}"),
                               html.Td(
                                   f"${(self.close[ticker][-1] * self.shares[ticker]) - self.positions[ticker]:,.2f}"),
                               html.Td(
                                   f"{((self.close[ticker][-1] * self.shares[ticker]) - self.positions[ticker]) / self.positions[ticker]:.2%}")
                           ]) for ticker in self.tickers
                       ], className="stats-table")
        ], style={"margin": "30px 0"})

        # Return the Div
        return div

    def _process_data(self) -> tuple:
        """
            Processes yfinance data for given tickers over a given range

            Returns:
                tuple of portfolio metrics as the result of preprocessing
        """
        # Get the ticker symbols
        tickers = list(self.positions.keys())

        # Get the data for the tickers for the last 5 years
        data = yf.download(tickers,
                           start=self.start_date,
                           end=datetime.datetime.now())

        # Get the close prices
        close = data["Close"]
        open_ = data['Open']
        low = data['Low']
        high = data['High']

        # Fill in missing data with the nearest previous value (and then backfill for the first values)
        close = close.fillna(method="ffill")
        close = close.fillna(method="bfill")

        if len(tickers) == 1:
            # Convert to a dataframe and rename the column to the ticker
            close = close.to_frame()
            close.columns = [tickers[0]]
            open_ = open_.to_frame()
            open_.columns = [tickers[0]]
            low = low.to_frame()
            low.columns = [tickers[0]]
            high = high.to_frame()
            high.columns = [tickers[0]]

        # Get the shares of each position
        shares = {ticker: self.positions[ticker] / close[ticker][0] for ticker in tickers}

        return close, open_, low, high, shares, data.index, tickers

    @staticmethod
    def _configure_layout(fig):
        """
            Internal function to configure portfolio potting layout given a certain time regimine and between chart types.

            Args:
                fig (plotly.go): Plotly instance of a figure to be configured
            
            Returns:
                Reconfigured object for display
        """
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date",
                fixedrange=True
            ),
            yaxis=dict(
                fixedrange=True
            ),
            margin=dict(t=0, b=0)
        )

        # Adds the option to toggle between a candle and line chart
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{'visible': [True, False]}],
                            label="Line",
                            method="update"
                        ),
                        dict(
                            args=[{'visible': [False, True]}],
                            label="Candle",
                            method="update"
                        )
                    ]),
                    direction='down',
                    pad={'r': 10, 't': 10},
                    x=0,
                    xanchor='left',
                    y=1.02,
                    yanchor='top',
                )])

        return fig


class TickerAutofill():

    def __init__(self) -> None:
        self.company_names_df: pd.DataFrame = pd.read_csv('data/NYSE_stock_tickers.csv')

    @staticmethod
    def jump_search(searching_array, val) -> int:
        """
            Python implementation of a jump search algorithm (to see if a given company name has a stock ticker)
        """
        length: int = len(searching_array)
        jump: int = int(math.sqrt(length))
        left, right = 0, 0
        while left < length and searching_array[left] <= val:
            right: int = min(length - 1, left + jump)
            if searching_array[left] <= val and searching_array[right] >= val:
                break
            left += jump
        if left >= length or searching_array[left] > val:
            return -1
        right = min(length - 1, right)
        i: int = left
        while i <= right and searching_array[i] <= val:
            if searching_array[i] == val:
                return i
            i += 1
        return -1

    def autocomplete_ticker(self, company_name) -> str:
        """
            Method for auto-complete suggestions of NYSE ticker symbols, given that ticker can be annoyingly confusing or random.
            Also will generate a given ticker symbol from the full typed company name.
        """
        company_names: list = [str(_).lower() for _ in list(self.company_names_df['CompanyName'])]

        removal_words: list[str] = ['inc.', ',', 'corp.', 'corporation', '  ']

        for item in removal_words:
            company_names: list = [_.replace(item, ' ') for _ in company_names]

        index: int = [index for index in range(len(company_names)) if company_name.lower() in company_names[index]][0]

        if index == -1:
            return company_name

        return str(self.company_names_df['StockTickers'][index]).upper()
