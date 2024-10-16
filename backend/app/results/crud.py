import aiofiles


class ResultsCRUD:
    @classmethod
    async def read_result_files(cls) -> list[str]:
        result_files = []
        results_dir = "results/"

        async for entry in aiofiles.os.scandir(results_dir):
            if entry.is_file():
                result_files.append(entry.name)

        return result_files
