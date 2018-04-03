import matplotlib.pyplot as plt
import pylab
from pylab import rcParams

import networkx as nx
import numpy as np

def createDirectedMap():
    '''建立简单有向图'''
    rcParams['figure.figsize']=10,10#设置graph为10*10英寸

    G=nx.DiGraph()
    #添加边和顶点
    G.add_edges_from([('K','I'),('R','T'),('V','T')],weight=3)
    G.add_edges_from([('T','K'),('T','H'),('I','T'),('T','H')],weight=4)
    G.add_edges_from([('I','R'),('H','N')],weight=5)
    G.add_edges_from([('R','N')],weight=6)
    #设置结点颜色
    val_map={'K':1.5,'I':0.9,'R':0.6,'T':0.2}
    values=[val_map.get(node,1.0) for node in G.nodes()]

    edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])

    #设置边的颜色
    red_edges=[('R','T'),('T','K')]
    edge_colors=['green' if not edge in red_edges else 'red' for edge in G.edges()]

    pos=nx.spring_layout(G)

    nx.draw_networkx_edges(G,pos,width=2.0,alpha=0.65,arrows=True)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    nx.draw(G,pos,node_color=values,node_size=1500,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
    pylab.show()

def simpleRoadMap():
    '''展示路线图'''
    G=nx.Graph(name='python')
    graph_routes=[[11,3,4,1,2],[5,6,3,0,1],[2,0,1,3,11,5]]
    edges=[]
    for r in graph_routes:
        route_edges=[(r[n],r[n+1]) for n in range(len(r)-1)]
        G.add_nodes_from(r)
        G.add_edges_from(route_edges)
        edges.append(route_edges)
    print('Graph has %d nodes with %d edges.'%(G.number_of_nodes(),G.number_of_edges()))
    pos=nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos=pos)
    nx.draw_networkx_labels(G,pos=pos)
    colors=['#00bb00','#4e86cc','y']
    linewidths=[22,14,10]
    for ctr,edgelist in enumerate(edges):
        nx.draw_networkx_edges(G,pos=pos,edgelist=edgelist,edge_color=colors[ctr],width=linewidths[ctr])
    pylab.show()

def simple_shortest_path():
    '''实现最短路径/最短距离'''
    g=nx.DiGraph()
    g.add_edge('m','i',weight=0.1)
    g.add_edge('i','a',weight=1.5)
    g.add_edge('m','a',weight=1.0)
    g.add_edge('a','e',weight=0.75)
    g.add_edge('e','h',weight=1.5)
    g.add_edge('a','h',weight=2.2)
    print(nx.shortest_path(g,'i','h'))
    edge_labels = dict([((u, v,), d['weight']) for u, v, d in g.edges(data=True)])
    pos = nx.spring_layout(g)
    nx.draw(g,pos=pos)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(g,pos=pos)
    pylab.show()

def les_miserables_related_people():
    '''读取《悲惨世界》相关联的人物'''
    G=nx.read_gml('lesmiserables.gml')
    G8=G.copy()
    dn=dict(nx.degree(G8))
    nodeList=list(G8.nodes())
    for n in nodeList:
        if dn[n]<=8:
            G8.remove_node(n)
    pos=nx.spring_layout(G8)
    nx.draw(G8,node_size=10,edge_color='b',alpha=0.45,font_size=9,pos=pos)
    labels=nx.draw_networkx_labels(G8,pos=pos)
    pylab.show()

if __name__=='__main__':
    #createDirectedMap()
    #simpleRoadMap()
    #simple_shortest_path()
    les_miserables_related_people()