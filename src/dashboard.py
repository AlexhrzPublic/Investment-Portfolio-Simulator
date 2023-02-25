from dash import dcc, html
from dash_extensions.enrich import Output, Input, State, ALL, MATCH, DashProxy, MultiplexerTransform  # conda install
from analyze_portfolio import PortfolioAnalyzer, TickerAutofill
import dash_bootstrap_components as dbc  # conda install
import yfinance as yf
import datetime

app = DashProxy(transforms=[MultiplexerTransform()])  # initializes app with multiplexer transforms
config = {
    'displayModeBar': False,
    'displaylogo': False,
    'scrollZoom': False,
}  # sets application scroll/display configs to False, want to work as a dashboard not a massive webpage

autocomplete_ticker: TickerAutofill = TickerAutofill()

# The layout of the dashboard is an arbitrary number of input fields,
# each takes the stock ticker as input and the dollar amount to invest
app.layout = html.Div([
    html.Div([
        html.Div(style={'height': '5px'}),
        html.Div([
            html.Div([
                html.H1('Stock Portfolio Simulator', style={'margin-left': '10px', 'color': 'navy'}),  # main app header
                html.H4('Please specify your stock portfolio below', style={'margin-left': '20px', 'color': 'black'}),
            ], style={'width': '80%', 'display': 'inline-block'}),
            html.Div([
                html.Div(style={'height': '10px'}),
                html.Div([
                    html.Div('Number of stocks: ', className='stock-number-label'),
                    # label and toggle for controlling the number of stocks in a portfolio
                    dcc.Input(id='stock-count', type='number', value=1,
                              min='1', max='100', className='stock-number-input'),
                ], className='submit-input-container'),
                html.Div([
                    html.Div('Start date: ', className='start-date-label'),
                    # calendar dropdown for selecting portfolio date of beginning
                    dbc.Input(id='start-date', type='date',
                              value=(datetime.datetime.now() - datetime.timedelta(days=365 * 5 + 1)).strftime(
                                  '%Y-%m-%d'), className='start-date-input'),
                ], className='submit-input-container'),  # final submit button to run analytics
                html.Div([
                    html.Button('Submit', id='submit-button', n_clicks=0, className='submit-button'),
                                # style={'background': 'darkgrey', 'border-radius': '16px', 'border': '1px'}),
                ])
            ], className='submit-container'),
        ], style={'width': '100%', 'display': 'flex'}),
        html.Div(style={'height': '15px'}),
        # div for each portfolio element, containing a unique ticker, amount purchased, and error display
        html.Div([
            html.Div([
                html.H3('Stock Ticker'),  # ticker input
                html.Div([], id='tickers')
            ], style={'width': '15%'}, className='input-column'),
            html.Div([
                html.H3('Dollar Amount ($)'),  # amount toggle
                html.Div([], id='amounts')
            ], style={'width': '20%'}, className='input-column'),
            html.Div([
                html.H3('Error Status'),  # error handler display
                html.Div([], id='errors')
            ], style={'width': '15%'}, className='input-column'),
            html.Div([
                html.H3('Info'),  # info display
                html.Div([], id='info_buttons')]
                , style={'width': '25%'}, className='input-column'),
            html.Div([
                html.H3('​'),
                html.Div([], id='delete_buttons')  # delete button for each stock in case of mistake
            ], style={'width': '3%', 'padding': '0px 20px 20px 0px'}, className='input-column'),
        ], style={'width': '100%', 'display': 'flex'}),
        html.Div(style={'height': '200px'}),
        html.Div([html.H5('Powered by Yahoo! Finance', style={'height': '10px', 'padding': '300px 800px 5px 800px'}),
                  html.Img(src=app.get_asset_url('yahoo.png'),
                           style={'height': '25px', 'display': 'block', 'margin-left': 'auto',
                                  'margin-right': 'auto'}), ],
                 # 'padding': '1px 750px 50px 750px'
                 style={'height': '200px', 'padding': '0px 0px 250px 0px'})
    ], style={'width': '100%', 'margin': '1px 1px 1px 1px', 'background-color': 'white', 'color': 'black'},
        id='portfolio-input'),
    html.Div([  # analysis portion of the dashboard, able to be toggeled on/off with submit button
        html.Div([
            dcc.Graph(id='portfolio-plot',
                      style={'height': '100%', 'width': '50%', 'background-color': 'white', 'color': 'black'},
                      config=config),  # portfolio plot element for overal portfolio performance
            dcc.Graph(id='individual-stock-plot',
                      style={'height': '100%', 'width': '50%', 'background-color': 'white', 'color': 'black'},
                      config=config),  # overlaid plot element for comparison of individual stock performance
        ], style={'width': '100%', 'height': '54vh', 'display': 'flex', 'padding-top': '4vh',
                  'background-color': 'white'}),
        html.Div([
            html.Div(style={'float': 'left', 'background-color': 'white', 'color': 'black'},
                     className='stats-container', id='portfolio-stats'),  # portfolio overall stats
            html.Div(style={'float': 'right', 'background-color': 'white', 'color': 'black'},
                     className='stats-container', id='stock-stats'),  # individual stock stats
        ], style={'width': '100%', 'height': '30vh', 'margin': '2vh 0'}),
        html.Center([
            html.Button('Edit portfolio', id='edit-button', n_clicks=0,
                        className='submit-button', style={'width': '200px', 'height': '4vh'})  # re-route to return and edit portfolio contents
        ])
    ], style={'width': '100%', 'height': '100vh', 'background-color': 'white', 'display': 'none'}, id='analysis')]
        #html.Div([
            #dcc.Graph(id='prediction-stock-plot', style={'height': '100%', 'width': '50%'}, config=config),
            #html.P(" Enter the prediction years:"),
            # dcc.Input(id='prediction-year', type='number',value=2018, min=1000, max=2022, step=1, className='prediction-year-input')
            #html.P("Select prediction mode:"),
            #dcc.Dropdown(id='prediction-mode', options=['Exponential prediction','Holt prediction'],
                         #value='Exponential prediction', clearable = False, className='prediction-mode-input')])
         )


# The callback function is called the number of stocks is changed
@app.callback(
    Output('tickers', 'children'),
    Output('amounts', 'children'),
    Output('errors', 'children'),
    Output('info_buttons', 'children'),
    Output('delete_buttons', 'children'),
    Input('stock-count', 'value'),
    State('tickers', 'children'),
    State('amounts', 'children'),
    State('errors', 'children'),
    State('info_buttons', 'children'),
    State('delete_buttons', 'children'), prevent_initial_call=False)
def add_stock(count, tickers, amounts, errors, infos, deletes):
    """
        Callback function for adding a stock to the given portfolio

        count: number of stocks to add to the portfolio
        tickers: given tickers of portfolio
        amounts: amount purchased of the given tickers
        errors: any error presiding over given tickers (invalid ticker)
        deleted: removed stocked from portfolio

    """
    if count is None:
        count = 1

    tickers = tickers[:count]
    amounts = amounts[:count]
    errors = errors[:count]
    infos = infos[:count]
    deletes = deletes[:count]

    for idx in range(len(tickers), count):
        tickers.append(html.Div([
            dcc.Input(id={
                'type': 'ticker',
                'index': idx,
            }, type='text', placeholder='Ticker or company', autoComplete='off', style={'text-transform': 'uppercase'}),
        ], style={'width': '20%'}, className='stock-input-div'))
        amounts.append(html.Div([
            dcc.Input(id={
                'type': 'amount',
                'index': idx,
            }, type='number', placeholder='Amount', step="0.01", min="0.01", autoComplete='off', value="1"),
        ], style={'width': '20%'}, className='stock-input-div'))
        errors.append(html.Div([
            html.Div(id={
                'type': 'error',
                'index': idx,
            })], style={'width': '75%', 'font-size': '15px'}, className='stock-input-div'))
        infos.append(html.Div([
            dbc.Button('i', id={
                'type': 'info-button',
                'index': idx
            }, n_clicks=0, className='info-button'),]+
            [dbc.Tooltip('Company or Crypto Information',
                        id={'type': 'info-tooltip', 'index':idx},
                        target={'type': 'info-button',
                                'index': idx},
                        placement='right',
                        style={
                        'display': 'inline-block',
                        'border': '1px dotted black',
                        'margin-left': '50px',
                        'margin-top': '50px',
                        'margin-bottom': '150px',
                        'border-radius':'15px',
                        'font-size': '14px',
                        'width':'300px',
                        'font-family': 'system-ui',
                        'background-color': 'lightgray',
                        'text-align': 'center',
                        'opacity':'0.7'}
                        )]
        , style={'width': '100%'}, className='stock-input-div')),
        deletes.append(html.Div([html.Div(style={'height': '8px'}),
                                 html.Button('×', id={
                                     'type': 'delete-button',
                                     'index': idx,
                                 }, n_clicks=0, className='delete-button')
                                 ], style={'width': '100%'}, className='stock-input-div'))

    return tickers, amounts, errors, infos, deletes


# The callback function is called when a stock is deleted
@app.callback(
    Output('tickers', 'children'),
    Output('amounts', 'children'),
    Output('errors', 'children'),
    Output('info_buttons', 'children'),
    Output('delete_buttons', 'children'),
    Output('stock-count', 'value'),
    Output({'type': 'delete-button', 'index': ALL}, 'n_clicks'),
    Input({'type': 'delete-button', 'index': ALL}, 'n_clicks'),
    State('tickers', 'children'),
    State('amounts', 'children'),
    State('errors', 'children'),
    State('info_buttons', 'children'),
    State('delete_buttons', 'children'),
    State('stock-count', 'value'), prevent_initial_call=True)
def delete_stock(clicks, tickers, amounts, errors, infos, deletes, count):
    """
        Callback function to remove a stock from the portfolio upon the action of the delete button

        clicks: number of stock to be deleted
        tickers: tickers for remaining stocks
        amounts: amounts for remaining stocks
        errors: errors presiding on the stocks
        deleted: delete buttons
        count: total stock count
    """
    for i in range(len(clicks)):
        if clicks[i] > 0:
            del tickers[i]
            del amounts[i]
            del errors[i]
            del infos[i]
            del deletes[i]
            count -= 1

    return tickers, amounts, errors, infos, deletes, count, [0] * len(clicks)


# The callback function is called when a stock is modified
@app.callback(Output({'type': 'error', 'index': MATCH}, 'children'),
              [Input({'type': 'ticker', 'index': MATCH}, 'value'),
               Input({'type': 'amount', 'index': MATCH}, 'value')], prevent_initial_call=False)
def update_error(ticker, amount):
    if ticker is None:
        return 'Missing or invalid ticker symbol.'

    if amount is None:
        return 'Dollar amount must have no more than 2 decimal places.'

    assert type(ticker) == str
    assert type(amount) in [int, float] or (type(amount) == str and amount.isnumeric()), type(amount)
    assert float(amount) > 0

    # Check if the ticker is valid
    if yf.Ticker(ticker).info['regularMarketPrice'] is None:
        return 'Ticker not present in our database.'

    return 'OK'


# Callback function associated with checking if a given company name is available
@app.callback(
    Output({'type': 'ticker', 'index': MATCH}, 'value'),
    Input({'type': 'ticker', 'index': MATCH}, 'value'), prevent_initial_call=True)
def check_name(company):
    """
        Callback function for adding auto-generated ticker names to the portfolio

        ticker: given tickers of portfolio
    """
    print(f'checking for companies named {company}')
    try:
        if yf.Ticker(company).info['regularMarketPrice'] is None:
            return autocomplete_ticker.autocomplete_ticker(company)
    except IndexError:
        pass

    return company


# callback for company hover information
@app.callback(
    Output({'type': 'info-tooltip', 'index': MATCH}, 'children'),
    Input({'type': 'ticker', 'index': MATCH}, 'value'), prevent_initial_call=True)
def tooltip_info(ticker):
    """
        Callback function for updating the company information tooltip.

        ticker: given tickers of portfolio
    """
    try:
        y = yf.Ticker(ticker).info['regularMarketPrice']
        if y != None:
            info = yf.Ticker(ticker).info
            return html.Div([html.H3(info['longName']), 
                            html.H4('Current Market Price:'), html.H5("$"+str(info['currentPrice'])),
                            html.H4('Sector:'), html.H5(info['sector']),
                            html.H4('Industry:'), html.H5(info['industry']),
                            html.H4('Full-Time Employees:'), html.H5(info["fullTimeEmployees"]),
                            html.H4('Company Overview'), html.H5(info["longBusinessSummary"][:info["longBusinessSummary"].index('.',100)+1])
                            ])
        else:
            return "Invalid Company Name"
    except:
        return "Company or Crypto Information"

# The callback function is called when the submit button is clicked
# It sets the display of the portfolio input to none and the analysis to initial
@app.callback(
    Output('portfolio-input', 'style'),
    Output('analysis', 'style'),
    Output('portfolio-input', 'children'),
    Output('portfolio-plot', 'figure'),
    Output('individual-stock-plot', 'figure'),
    # Output('predicted-stock-plot', 'figure'),
    Output('portfolio-stats', 'children'),
    Output('stock-stats', 'children'),
    # Input('prediction-model','value'),
    # Input('prediction-year','value')
    Input('submit-button', 'n_clicks'),
    State('portfolio-input', 'style'),
    State('analysis', 'style'),
    State('errors', 'children'),
    State('portfolio-input', 'children'),
    State('tickers', 'children'),
    State('amounts', 'children'),
    State('start-date', 'value'), prevent_initial_call=True)
def submit_portfolio(_, portfolio_style, analysis_style, statuses, children, tickers, amounts, start_date):
    error = False
    for status in statuses:
        if status['props']['children'][0]['props']['children'] != 'OK':
            error = True
            break

    if error or len(statuses) == 0:
        children = [dbc.Alert("Please fix all errors before submitting",
                              duration=4000, fade=True, className='error-alert')] + children
        return portfolio_style, analysis_style, children, {}, {}, {}, {}

    # collect the data from the input fields
    positions = {}
    for i in range(len(tickers)):
        ticker = tickers[i]['props']['children'][0]['props']['value'].upper().strip()
        amount = amounts[i]['props']['children'][0]['props']['value']
        if ticker not in positions:
            positions[ticker] = float(amount)
        else:
            positions[ticker] += float(amount)

    # set the display of the portfolio input to none and the analysis to initial
    portfolio_style['display'] = 'none'
    analysis_style['display'] = 'initial'
    analysis_style['background-color'] = 'white'
    analysis_style['color'] = 'black'

    # run the analysis
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    stock_analyser = PortfolioAnalyzer(positions, start_date)

    portfolio_graph = stock_analyser.graph_portfolio()
    individual_graph = stock_analyser.graph_individual_stocks()
    portfolio_stats = stock_analyser.get_portfolio_stats()
    stock_stats = stock_analyser.get_stock_stats()
    #if prediction_mode == 'Exponential prediction':
        #prediction_graph = stock_analyser.exponential_smoothing()
    #else:
        #prediction_graph = stock_analyser.holt_smoothing()
    return portfolio_style, analysis_style, children, portfolio_graph, individual_graph, portfolio_stats, stock_stats


# The callback function is called when the edit button is clicked
# It sets the display of the portfolio input to initial and the analysis to none
@app.callback(
    Output('portfolio-input', 'style'),
    Output('analysis', 'style'),
    Input('edit-button', 'n_clicks'),
    State('portfolio-input', 'style'),
    State('analysis', 'style'), prevent_initial_call=True)
def edit_portfolio(_, portfolio_style, analysis_style):
    portfolio_style['display'] = 'initial'
    portfolio_style['background-color'] = 'white'
    portfolio_style['color'] = 'black'
    analysis_style['display'] = 'none'

    return portfolio_style, analysis_style


if __name__ == '__main__':
    app.run_server(debug=True)
