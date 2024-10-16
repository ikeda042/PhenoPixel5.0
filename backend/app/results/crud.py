import os
import asyncio


class ResultsCRUD:
    @classmethod
    async def read_result_files(cls) -> list[str]:
        result_files = []
        results_dir = "results/"

        loop = asyncio.get_event_loop()
        entries = await loop.run_in_executor(None, os.scandir, results_dir)

        for entry in entries:
            if entry.is_file():
                result_files.append(entry.name)

        return [
            file
            for file in result_files
            if file.split(".")[-1] not in ["py", "pyc", "gitignore"]
        ]
