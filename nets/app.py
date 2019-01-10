# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from logistic_regr_xrp import XrpDataHandler
from my_network import VectorizedNet
import plotly.graph_objs as go

class Aplication:

    app = None
    data_handler = None
    dp = None
    net = None

    table_name = 'xrp'
    styles = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=self.styles)
        self.data_handler = XrpDataHandler('xrp', 'close')
        self.net = VectorizedNet(input_size=50, trainig_sets=10000, num_iterations=2000, learning_rate=0.3)
        self.draw_layout()

        @self.app.callback(
            Output(component_id='graph', component_property='figure'),
            [Input(component_id='options', component_property='value'), Input(component_id='mav-periods', component_property='value')]
        )
        def draw_graph(graph_type, periods):
            datax, datay = self.data_handler.get_timerow('close')
            data_rows = go.Scatter(x=datax, y=datay, mode='lines')
            data_rows.name = graph_type
            try:
                periods = int(periods)
            except:
                periods = 10
            mav = go.Scatter(x=datax, y=self.data_handler.get_ma(periods), mode='lines')
            return {'data': [data_rows, mav]}

        self.app.run_server(debug = True)

    def get_h4_header(self, header_text):
        return html.H4(children=header_text, style={'text-align':'center'})

    def get_radiobuttons(self, opts):
        return html.Div(
                    children=
                        dcc.RadioItems(
                            id='options',
                            options=[{'label': option, 'value': option} for option in opts],
                            value='open'
                            ),
                    style={'display': 'table', 'text-align': 'center', 'border': '1px solid red', 'height':'50%', 'width':'100%'}
        )

    def get_ma_input(self):
        return html.Div(
                    children=[
                        html.Div(children='MA: ', style={'display': 'table-cell'}),
                        dcc.Input(
                            id='mav-periods',
                            style={'display': 'table-cell', 'width':'70%', 'margin-left': '1%'},
                            value=10
                        ),
                    ],
                    style={'display': 'table', 'height':'50%', 'border': '1px solid green', 'horizontal-align':'middle'}
        )

    def draw_layout(self):
        self.app.layout = html.Div(
                                children=[
                                    self.get_h4_header('XRP/USD'),
                                    html.Div(
                                        children=[
                                            html.Div(
                                                children=
                                                    dcc.Graph(id='graph'),
                                                    style={'display': 'table-cell', 'width': '95%', 'border': '1px solid blue'}
                                                    ),
                                                    html.Div(
                                                        children=[
                                                            self.get_radiobuttons(['open', 'close']),
                                                            self.get_ma_input()
                                                        ],
                                                        style={'display': 'table-cell', 'border': '1px solid green', 'vertical-align':'middle'}
                                                    )
                                        ],
                                        style={'text-align':'center', 'border': '1px solid black', 'display': 'table', 'width':'100%', 'height': '100%'}
                                        )
                                ]
        )


if __name__ == '__main__':
    app = Aplication()