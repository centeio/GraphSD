import matplotlib.pyplot as plt
import networkx as nx

def graphViz(graph, width=1):
    ecolors = list(nx.get_edge_attributes(graph,'weight').values())
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, edge_color=ecolors,
            width=4, edge_cmap=plt.cm.Blues, with_labels=True, cmap=plt.cm.Reds, figsize=(20,20))

    plt.figure(figsize=(5,5)) 
    plt.show()

def displayGender(graph, ids, socialData, shifth = 2, shiftv = 0, filtere=0, width = 1):
    
    #TODO debug filter edges
    #if filtere != 0:
    #   graph = graph.edge_subgraph(filterEdges(graph, filtere))
    #   print(list(graph.edges(data=True)))

    subgraphF = graph.subgraph([nid for nid in ids if (socialData.query("id==@nid").Gender.item() == 'F')])
    subgraphM = graph.subgraph([nid for nid in ids if (socialData.query("id==@nid").Gender.item() == 'M')])

    posF = nx.circular_layout(subgraphF)
    posM = nx.circular_layout(subgraphM)

    for val in posF:
            posF[val][0] = posF[val][0] - shifth
            posF[val][1] = posF[val][1] - shiftv
    for val in posM:
            posM[val][0] = posM[val][0] + shifth
            posM[val][1] = posM[val][1] + shiftv
            
    pos = {**posF, **posM}

    ecolors = list(nx.get_edge_attributes(graph,'weight').values())
    ncolors = [0.4 if socialData.query("id==@nid").Gender.item() == 'F' else 0.5 for nid in ids]

    nx.draw(graph, pos, node_color=ncolors, edge_color = ecolors,
        width=width, edge_cmap=plt.cm.Blues, with_labels=True, cmap=plt.cm.Reds, figsize=(20,20))

    plt.show()

def attHist(attribute, value, bins = 5):
    listedges = []
    wsum = 0
    count = 0

    for nid1, nid2, edict in list(GComp.edges(data=True)):
        if edict[attribute] == value:
            listedges += [edict['weight']] 
            wsum += edict['weight']
            count += 1
            
            
    print('mean: ', wsum/count)
            
            
    listedges
    plt.hist(listedges, bins)

def printpositions(dataset, ids, initialDate, finalDate): #colors in RGB 0-255
    #cmapb = plt.cm.Blues
    #cmapb = cmapb(list(nx.get_edge_attributes(G,'weight').values()))
    cmapr = plt.cm.Reds
    weights = list(range(len(ids)))
    max_id = max(weights)
    rangec = [x/max_id for x in weights]
    ncolor = cmapr(rangec)
    
    dates = []

    start_window = pd.Timestamp(initialDate)
    nseconds = 1

    while start_window <= pd.Timestamp(finalDate):
        dates += [str(start_window)]
        #coords = after18[str(start_window)][['id','x','y']].set_index("id").reindex(ids).reset_index()[['x','y']]
        #compute distances
        # np.nan_to_num(dists)
        start_window = start_window + pd.Timedelta(seconds = nseconds)

    start_window = pd.Timestamp(initialDate)
    
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }

    # fill in most of layout
    #figure['layout']['xaxis'] = {'range': [30, 85], 'title': 'X'}
    figure['layout']['xaxis'] = {'title': 'X'}
    figure['layout']['yaxis'] = {'title': 'Y'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['sliders'] = {
        'args': [
            'transition', {
                'duration': 400,
                'easing': 'cubic-in-out'
            }
        ],
        'initialValue': initialDate,
        'plotlycommand': 'animate',
        'values': dates,
        'visible': True
    }
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': False},
                             'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                    'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 10},
            'prefix': 'Timestamp:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # make data
    colcounter = 0
    for nid in ids:
        dataset_by_ts = dataset[str(start_window)].set_index("id").reindex(ids).reset_index()
        dataset_by_ts_and_id = dataset_by_ts[dataset_by_ts['id'] == nid]


        data_dict = {
            'x': list(dataset_by_ts_and_id['x']),
            'y': list(dataset_by_ts_and_id['y']),
            'mode': 'markers',
            'text': list(dataset_by_ts_and_id['id']),
            'marker': {
                'sizemode': 'area',
                'sizeref': 200000,
                'size': 10,
                'color': 'rgb('+ str(round(ncolor[colcounter][0]*255)) + ',' + str(round(ncolor[colcounter][1]*255)) + ',' + str(round(ncolor[colcounter][2]*255)) + ')'
            },
            'name': nid
        }
        figure['data'].append(data_dict)
        colcounter += 1

    # make frames
    for ts in dates:
        frame = {'data': [], 'name': str(ts)}
        
        colcounter = 0
        for nid in ids:
            dataset_by_ts = dataset[ts].set_index("id").reindex(ids).reset_index()
            dataset_by_ts_and_id = dataset_by_ts[dataset_by_ts['id'] == nid]

            data_dict = {
                'x': list(dataset_by_ts_and_id['x']),
                'y': list(dataset_by_ts_and_id['y']),
                'mode': 'markers',
                'text': list(dataset_by_ts_and_id['id']),
                'marker': {
                    'sizemode': 'area',
                    'sizeref': 200000,
                    'size': 10,
                    'color': 'rgb('+ str(round(ncolor[colcounter][0]*255)) + ',' + str(round(ncolor[colcounter][1]*255)) + ',' + str(round(ncolor[colcounter][2]*255)) + ')'
                },
                'name': nid
            }
            frame['data'].append(data_dict)
            colcounter += 1

        figure['frames'].append(frame)
        slider_step = {'args': [
            [ts],
            {'frame': {'duration': 200, 'redraw': False},
             'mode': 'immediate',
           'transition': {'duration': 200}}
         ],
         'label': ts,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)
        


    figure['layout']['sliders'] = [sliders_dict]

    iplot(figure)

def printpositionsG(G, initialDate, finalDate): #colors in RGB 0-255
    cmapb = plt.cm.Blues
    cmapb = cmapb(list((socialData.set_index("id").reindex(ids).Gender == 'F') + 0))
    gender = socialData.set_index("id").reindex(ids).Gender

    dates = []

    start_window = pd.Timestamp(initialDate)
    nseconds = 1

    while start_window <= pd.Timestamp(finalDate):
        dates += [str(start_window)]
        #coords = after18[str(start_window)][['id','x','y']].set_index("id").reindex(ids).reset_index()[['x','y']]
        #compute distances
        # np.nan_to_num(dists)
        start_window = start_window + pd.Timedelta(seconds = nseconds)


    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }

    # fill in most of layout
    #figure['layout']['xaxis'] = {'range': [30, 85], 'title': 'X'}
    figure['layout']['xaxis'] = {'title': 'X'}
    figure['layout']['yaxis'] = {'title': 'Y'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['sliders'] = {
        'args': [
            'transition', {
                'duration': 400,
                'easing': 'cubic-in-out'
            }
        ],
        'initialValue': initialDate,
        'plotlycommand': 'animate',
        'values': dates,
        'visible': True
    }
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': False},
                             'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                    'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 10},
            'prefix': 'Timestamp:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # make data
    colcounter = 0
    for nid in ids:
        dataset_by_ts = after18[str(start_window)].set_index("id").reindex(ids).reset_index()
        dataset_by_ts_and_id = dataset_by_ts[dataset_by_ts['id'] == nid]


        data_dict = {
            'x': list(dataset_by_ts_and_id['x']),
            'y': list(dataset_by_ts_and_id['y']),
            'mode': 'markers',
            'text': gender[nid],
            'marker': {
                'sizemode': 'area',
                'sizeref': 200000,
                'size': 10,
                'color': 'rgb('+ str(round(colors[colcounter][0]*255)) + ',' + str(round(colors[colcounter][1]*255)) + ',' + str(round(colors[colcounter][2]*255)) + ')'
            },
            'name': nid
        }
        figure['data'].append(data_dict)
        colcounter += 1

    # make frames
    for ts in dates:
        frame = {'data': [], 'name': str(ts)}
        
        colcounter = 0
        for nid in ids:
            dataset_by_ts = after18[ts].set_index("id").reindex(ids).reset_index()
            dataset_by_ts_and_id = dataset_by_ts[dataset_by_ts['id'] == nid]

            data_dict = {
                'x': list(dataset_by_ts_and_id['x']),
                'y': list(dataset_by_ts_and_id['y']),
                'mode': 'markers',
                'text': gender[nid],
                'marker': {
                    'sizemode': 'area',
                    'sizeref': 200000,
                    'size': 10,
                    'color': 'rgb('+ str(round(colors[colcounter][0]*255)) + ',' + str(round(colors[colcounter][1]*255)) + ',' + str(round(colors[colcounter][2]*255)) + ')'
                },
                'name': nid
            }
            frame['data'].append(data_dict)
            colcounter += 1

        figure['frames'].append(frame)
        slider_step = {'args': [
            [ts],
            {'frame': {'duration': 50, 'redraw': False},
             'mode': 'immediate',
           'transition': {'duration': 1}}
         ],
         'label': ts,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)
        


    figure['layout']['sliders'] = [sliders_dict]

    iplot(figure)