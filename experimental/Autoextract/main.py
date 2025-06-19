from cell_connector import CellConnector, CellBaseModel




for cell in CellConnector.get_cells():
    print(cell.cell_id, cell.label_experiment, cell.manual_label, cell.area)
