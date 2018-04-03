import igraph as ig
def simple_1():
    vertices=['A','B','C','D','E','F','G','H','I','J']#顶点
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,1),(1,8),(8,2),(2,4),(4,9),(9,5),(5,7),(7,0)]
    graphStyle={'vertex_size':20}
    g=ig.Graph(vertex_attrs={"label":vertices},edges=edges,directed=True)
    g.write_svg("simple_star.svg",width=500,height=300,**graphStyle)

def simple_read_net():
    '''读取pajek格式文件'''
    g=ig.read("testdata/GR3_60.NET",format="pajek")
    # 设置边和顶点颜色
    g.vs["color"]="#3d679d"
    g.es["color"]="red"

    graphStyle={"vertex_size":12,'margin':6}
    graphStyle["layout"]=g.layout("fr")#设置布局
    g.write_svg('GR3_60_graph.svg',width=600,height=600,**graphStyle)

def protein_interaction_network():
    g = ig.read("yeast/YeastS.net", format="pajek")
    #g.vs["color"] = "#3d679d"
    #g.es["color"] = "red"
    graphStyle = {"layout":'auto'}
    g.write_svg('YeastS_graph.svg', width=600, height=600,**graphStyle)
if __name__=='__main__':
    simple_1()
    simple_read_net()
    protein_interaction_network()