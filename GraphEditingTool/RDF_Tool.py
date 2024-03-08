import random

input_path = "CustomGraph.ttl"
out_path = "out.ttl"

import sys
import rdflib
import networkx as nx
from rdflib.namespace import RDF
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, \
    QGraphicsTextItem, QGraphicsLineItem, QPushButton, QInputDialog, QLabel
from PyQt5.QtCore import Qt, QLineF, QPointF
from PyQt5.QtGui import QBrush, QColor, QPen, QFont, QPainterPath, QTransform, QPalette

NODE_COLORS = [QColor('#8dd3c7'), QColor('#feffb3'), QColor('#bfbbd9'), QColor('#fa8174'), QColor('#81b1d2'),
               QColor('#fdb462'), QColor('#b3de69'), QColor('#bc82bd'), QColor('#ccebc4'), QColor('#ffed6f')]
NODE_DIAMETER = 100

sc_prefix = 'http://purl.org/science/owl/sciencecommons/'
prefix = sc_prefix
type_prefix = 'http://www.w3.org/1999/02/'


def configure_application(app):
    """
    Apply dark theme and style configuration to the PyQt application.

    Args:
    app (QApplication): The application to configure.
    """
    app.setStyle("Fusion")
    dark_palette = QPalette()

    # Base color for window background
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.black)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)

    dark_palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)
    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")



def generate_color_map(node_types):
    """
    Generates a color map for node types.

    Args:
    node_types (dict): A dictionary mapping nodes to their types.

    Returns:
    dict: A dictionary mapping node types to colors.
    """
    unique_types = set(node_types.values())
    color_map = {}
    color_iter = iter(NODE_COLORS)
    for node_type in unique_types:
        color_map[node_type] = next(color_iter, QColor(255, 255, 255))
    return color_map


def read_rdf_graph(file_path):
    """
    Reads an RDF graph from a file and creates a NetworkX graph.

    Args:
    file_path (str): Path to the RDF file.

    Returns:
    tuple: rdflib Graph, NetworkX graph, node types, and color map.
    """
    try:
        g = rdflib.Graph()
        g.parse(file_path)
        node_types = {str(node): "default" for node in g.all_nodes()}
        for subj, pred, obj in g:
            if pred == RDF.type:
                node_types[str(subj)] = str(obj)
        color_map = generate_color_map(node_types)
        return g, generate_nx_graph(g, color_map, node_types), node_types, color_map
    except Exception as e:
        print(f"Error reading RDF graph: {e}")
        sys.exit(1)


def generate_nx_graph(rdf_graph, color_map, node_types):
    """
    Generates a NetworkX graph from an RDF graph.

    Args:
    rdf_graph (rdflib.Graph): The RDF graph.
    color_map (dict): A color map for node types.
    node_types (dict): Node types.

    Returns:
    tuple: A NetworkX graph and edge labels.
    """
    G = nx.DiGraph()
    edge_labels = {}
    for subj, pred, obj in rdf_graph:
        subj_label = str(subj).split('/')[-1]
        obj_label = str(obj).split('/')[-1]
        G.add_node(subj_label, color=color_map[node_types[str(subj)]])
        G.add_node(obj_label, color=color_map[node_types[str(obj)]])
        G.add_edge(subj_label, obj_label)
        edge_labels[(subj_label, obj_label)] = pred.split('/')[-1]
    return G, edge_labels


# Custom QGraphicsItem for nodes
class GraphNode(QGraphicsEllipseItem):
    def __init__(self, x, y, diameter, label, scene, color, window):
        super().__init__(-diameter / 2, -diameter / 2, diameter, diameter)
        self.setBrush(QBrush(color))
        self.window = window
        self.setPen(QPen(Qt.NoPen))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable)
        self.setPos(x, y)
        self.scene = scene
        self.edges = []

        self.org_label = label
        # Split label if it's too long
        split_label = self.split_label_text(label, diameter)

        # Add text label
        self.label = QGraphicsTextItem(split_label, self)
        self.label.setFont(QFont("Arial", 10, QFont.Bold))
        # Calculate the text offset to center it
        label_rect = self.label.boundingRect()
        text_offset_x = (diameter - label_rect.width()) / 2 - diameter / 2
        self.label.setPos(text_offset_x, -label_rect.height() / 2)

    def split_label_text(self, label, diameter):
        # Logic to split the label text
        max_chars_in_line = int(diameter / 10)  # Estimate max characters per line
        split_label = label
        if len(label) > max_chars_in_line:
            firstpart, secondpart = label[:len(label) // 2], label[len(label) // 2:]
            split_label = firstpart + "-\n" + secondpart
        return split_label.strip()

    def mousePressEvent(self, event):

        self.scene.clearSelection()
        self.setSelected(True)
        super().mousePressEvent(event)
        if self.window.adding_edge:
            selected_node = self.window.get_selected_node()
            if selected_node:
                if selected_node != self.window.first_node_selected:
                    self.window.create_edge(self.window.first_node_selected, selected_node)
                    self.window.adding_edge = False
                    self.window.instructionLabel.setText("")

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        for edge in self.edges:
            edge.adjust()


## Custom QGraphicsLineItem for edges
class GraphEdge(QGraphicsLineItem):
    def __init__(self, source, target, label):
        super().__init__()
        self.source = source
        self.target = target
        self.label = QGraphicsTextItem(label, self)
        self.label.setDefaultTextColor(Qt.white)
        self.setPen(QPen(Qt.gray, 3))
        self.setFlag(QGraphicsLineItem.ItemIsSelectable)
        self.source.edges.append(self)  # Add edge to source node's edge list
        self.target.edges.append(self)  # Add edge to target node's edge list
        self.adjust()

    def adjust(self):
        if not self.source or not self.target:
            return

        line = QLineF(self.source.scenePos(), self.target.scenePos())
        self.prepareGeometryChange()
        self.setLine(line)

        # Update position of label
        mid_point = line.pointAt(0.5)
        self.label.setPos(mid_point - QPointF(self.label.boundingRect().width() / 2,
                                              self.label.boundingRect().height() / 2))

    def update_position(self):
        mid_point = QLineF(self.source.pos(), self.target.pos()).pointAt(0.5)
        self.label.setPos(mid_point)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

    def paint(self, painter, option, widget=None):
        if not self.source or not self.target:
            return

        # Get positions of the source and target
        source_pos = self.source.scenePos()
        target_pos = self.target.scenePos()

        # Create the line between source and target
        line = QLineF(source_pos, target_pos)

        # Shorten the line so the arrow does not overlap with the node
        node_radius = self.source.rect().width() / 2  # Assuming nodes are circles
        shorten_length = node_radius + 10  # 10 is the length of the arrow
        if line.length() > shorten_length:
            line.setLength(line.length() - shorten_length)

        painter.setPen(self.pen())
        painter.drawLine(line)

        # Calculate the angle of the line
        angle = line.angle()

        # Create an arrowhead
        arrow_size = 10.0
        arrow_head = QPainterPath()
        arrow_head.moveTo(0, 0)
        arrow_head.lineTo(-arrow_size, arrow_size)
        arrow_head.lineTo(arrow_size, arrow_size)
        arrow_head.lineTo(0, 0)
        painter.setBrush(Qt.gray)

        # Apply rotation and translation to the arrowhead
        transform = QTransform()
        transform.translate(line.p2().x(), line.p2().y())
        transform.rotate(-angle + 90)
        transformed_arrow = transform.map(arrow_head)

        painter.drawPath(transformed_arrow)

        # Update label position
        mid_point = line.pointAt(0.5)
        self.label.setPos(mid_point - QPointF(self.label.boundingRect().width() / 2,
                                              self.label.boundingRect().height() / 2))


# PyQt5 Main Window
class GraphWindow(QMainWindow):
    def __init__(self, rdf_graph, nx_graph, edge_labels):
        super().__init__()
        self.rdf_graph = rdf_graph
        self.nx_graph = nx_graph
        self.edge_labels = edge_labels
        self.node_items = {}
        self.first_node_selected = None
        self.adding_edge = False
        self.initUI()
        self.showMaximized()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('RDF Graph Visualization')

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)

        self.draw_graph()

        # Add 'Copy' button
        self.copyButton = QPushButton('Copy Node', self)
        self.copyButton.clicked.connect(self.copy_node)
        self.copyButton.move(10, 10)  # Position the button

        # Add 'Delete' button
        self.deleteButton = QPushButton('Delete Edge', self)
        self.deleteButton.clicked.connect(self.delete_edge)
        self.deleteButton.move(10, 40)  # Position the button
        # Add 'Add Edge' button
        self.addEdgeButton = QPushButton('Add Edge', self)
        self.addEdgeButton.clicked.connect(self.start_add_edge)
        self.addEdgeButton.move(10, 70)  # Position the button

        # Instruction label
        self.instructionLabel = QLabel(self)
        self.instructionLabel.move(10, 100)
        self.instructionLabel.resize(300, 20)

        self.addNodeButton = QPushButton('Add Node', self)
        self.addNodeButton.clicked.connect(self.add_node)
        self.addNodeButton.move(10, 100)  # Adjust position as needed

    def add_node(self):
        text, ok = QInputDialog.getText(self, 'Add Node', 'Enter new node name:')
        if ok and text:
            self.create_node(text)

    def create_node(self, node_name):
        # Create a new node at a default position
        x, y = 100, 100  # You might want to calculate these positions
        color = random.choice(NODE_COLORS)

        new_node = GraphNode(x, y, NODE_DIAMETER, node_name, self.scene, color, self)
        new_node.setPen(QPen(Qt.NoPen))
        self.scene.addItem(new_node)
        self.node_items[node_name] = new_node

        # Update RDF graph (Assuming default type for new nodes)
        node_uri = rdflib.URIRef(prefix+node_name)
        print(node_uri)
        self.rdf_graph.add((node_uri, RDF.type, rdflib.URIRef("defaultType")))  # Adjust 'defaultType' as needed
        #self.save_graph()

    def copy_node(self):
        selected_items = [item for item in self.scene.selectedItems() if isinstance(item, GraphNode)]
        if selected_items:
            original_node = selected_items[0]
            text, ok = QInputDialog.getText(self, 'Copy Node', 'Enter new node name:')
            if ok and text:
                self.create_copy_of_node(original_node, text)

    def create_copy_of_node(self, original_node, new_name):
        # Calculate position for the new node
        x, y = original_node.pos().x() + 20, original_node.pos().y() + 20  # Slight offset from original

        color = original_node.brush().color()

        # Create and add the new node
        new_node = GraphNode(x, y, 100, new_name, self.scene, color, self)
        new_node.setPen(QPen(Qt.NoPen))

        self.node_items[new_name] = new_node

        # Copy connections and update RDF graph
        for edge in original_node.edges:
            other_node = edge.source if edge.target == original_node else edge.target
            new_edge = GraphEdge(new_node, other_node, edge.label.toPlainText())
            self.scene.addItem(new_edge)

            # Update RDF graph
            source_label = prefix + new_name
            target_label = prefix + other_node.org_label
            edge_label = edge.label.toPlainText()
            pre = type_prefix if edge_label == '22-rdf-syntax-ns#type' else prefix
            self.rdf_graph.add(
                (rdflib.URIRef(source_label), rdflib.URIRef(pre + edge_label), rdflib.URIRef(target_label)))

        self.scene.addItem(new_node)

        self.save_graph()  # Save changes to the RDF graph

    def draw_graph(self):
        pos = nx.kamada_kawai_layout(self.nx_graph)
        # Create nodes
        for node, p in pos.items():
            x, y = p[0] * 1200, p[1] * 700  # Scale positions
            color = self.nx_graph.nodes[node]['color']
            node_item = GraphNode(x, y, 100, node, self.scene, color, self)  # Increased diameter
            self.node_items[node] = node_item

        # Create edges
        for edge in self.nx_graph.edges():
            source_item = self.node_items[edge[0]]
            target_item = self.node_items[edge[1]]
            label = self.edge_labels[edge]
            edge_item = GraphEdge(source_item, target_item, label)
            self.scene.addItem(edge_item)

        # Add nodes to the scene after creating edges
        for node_item in self.node_items.values():
            self.scene.addItem(node_item)

    def delete_edge(self):
        selected_items = [item for item in self.scene.selectedItems() if isinstance(item, GraphEdge)]
        if selected_items:
            edge_to_delete = selected_items[0]
            self.scene.removeItem(edge_to_delete)
            edge_to_delete.source.edges.remove(edge_to_delete)
            edge_to_delete.target.edges.remove(edge_to_delete)

            sub = rdflib.URIRef(prefix + edge_to_delete.source.org_label)
            edge_label = edge_to_delete.label.toPlainText()
            pre = type_prefix if edge_label == '22-rdf-syntax-ns#type' else prefix
            p = rdflib.URIRef(pre + edge_label)
            obj = rdflib.URIRef(prefix + edge_to_delete.target.org_label)

            self.rdf_graph.remove((sub, p, obj))
            self.save_graph()

    def start_add_edge(self):
        self.adding_edge = True
        self.first_node_selected = self.get_selected_node()
        self.instructionLabel.setText("Select the second node for the edge.")

    def get_selected_node(self):
        selected_items = [item for item in self.scene.selectedItems() if isinstance(item, GraphNode)]
        return selected_items[0] if selected_items else None

    def create_edge(self, source_node, target_node):
        edge_name, ok = QInputDialog.getText(self, 'Edge Name', 'Enter the name of the edge:')
        if ok and edge_name:
            new_edge = GraphEdge(source_node, target_node, label=edge_name)
            self.scene.addItem(new_edge)
            # Optionally, use edge_name somewhere in your application
            source_label = prefix + source_node.org_label
            target_label = prefix + target_node.org_label

            pre = type_prefix if edge_name == '22-rdf-syntax-ns#type' else prefix

            self.rdf_graph.add(
                (rdflib.URIRef(source_label), rdflib.URIRef(pre + edge_name), rdflib.URIRef(target_label)))
            self.save_graph()

    def save_graph(self):
        self.rdf_graph.serialize(destination=out_path, format='turtle')


# Main application
def main():
    rdf_graph, (nx_graph, edge_labels), node_types, color_map = read_rdf_graph(input_path)
    app = QApplication(sys.argv)
    configure_application(app)
    mainWindow = GraphWindow(rdf_graph, nx_graph, edge_labels)
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
