from backend.app.CellDBConsole.crud import CellCrudBase


async def main(db_name: str):
    crud: CellCrudBase = CellCrudBase(db_name)
    await crud.get_peak_paths_csv()


if __name__ == "__main__":
    import asyncio

    bd_name = "test_database.db"
    asyncio.run(main(bd_name))
