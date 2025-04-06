import aiohttp
import asyncio
import os


async def fetch_heatmap_data(session, db_name, cell_id=""):
    url = f"http://localhost:8000/api/cells/{db_name}/1/F0C16/heatmap/bulk/csv"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.text()
            return db_name, data
        else:
            return db_name, None


async def fetch_bulk_heatmap_data(db_names, cell_id):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for db_name in db_names:
            print(f"Fetching data for {db_name}...")
            tasks.append(fetch_heatmap_data(session, db_name, cell_id))
        results = await asyncio.gather(*tasks)
        return results


def run_fetch_bulk_heatmap_data(db_names, cell_id):
    return asyncio.run(fetch_bulk_heatmap_data(db_names, cell_id))


db_names = [
    i
    for i in os.listdir("backend/app/databases")
    if i.endswith(".db") and i != "test_database.db"
]
print(db_names)

results = run_fetch_bulk_heatmap_data(db_names, "")

# 結果の表示
for db_name, data in results:
    if data:
        print(f"Data for {db_name}:\n{data}")
    else:
        print(f"Failed to fetch data for {db_name}")
