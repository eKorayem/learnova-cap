from .providers import QdrantDBProvider
from .VectorDBEnums import VectorDBEnums
from controllers.BaseController import BaseController

class VectorDBProviderFactory:

    def __init__(self, config):
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider: str):

        if provider == VectorDBEnums.QDRANT.value:
            # Prefer URL for Docker/production, fall back to file path for local dev
            url = getattr(self.config, 'VECTOR_DB_URL', None)
            db_path = None

            if not url:
                db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_PATH)

            return QdrantDBProvider(
                url=url,
                db_path=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD
            )

        return None