import aiohttp
import asyncio


async def fetch_heatmap_data(session, db_name, cell_id=""):
    url = f"http://localhost:8000/api/cells/{db_name}/1/{cell_id}/heatmap/bulk/csv"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.text()
            return db_name, data
        else:
            return db_name, None  #


async def fetch_bulk_heatmap_data(db_names, cell_id):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for db_name in db_names:
            tasks.append(fetch_heatmap_data(session, db_name, cell_id))
        results = await asyncio.gather(*tasks)
        return results


def run_fetch_bulk_heatmap_data(db_names, cell_id):
    return asyncio.run(fetch_bulk_heatmap_data(db_names, cell_id))


import os

db_names = os.listdir("experimental/bulk-peakpaths-local")
print(db_names)
# cell_id = "12345"
# results = run_fetch_bulk_heatmap_data(db_names, cell_id)

# # 結果の表示
# for db_name, data in results:
#     if data:
#         print(f"Data for {db_name}:\n{data}")
#     else:
#         print(f"Failed to fetch data for {db_name}")
