class CellNotFoundError(Exception):
    """Exception raised when a cell is not found in the database."""

    def __init__(self, cell_id: str, message: str = "Cell not found"):
        self.cell_id = cell_id
        self.message = message
        super().__init__(f"{message}: {cell_id}")
