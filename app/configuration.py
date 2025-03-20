import os


class Config:
    POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
    POSTGRES_DBNAME = os.getenv('POSTGRES_DBNAME', 'vkr')
    POSTGRES_HOST = os.getenv(
        'POSTGRES_HOST', 'localhost'
    )
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')

    DATABASE_URI = f'postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DBNAME}'

    def data_for_psycopg_conn(self):
        """
        Функция составляет словарь для удобного подключения к psycopg2
        """
        data = {
            "dbname": self.POSTGRES_DBNAME,
            "user": self.POSTGRES_USERNAME,
            "password": self.POSTGRES_PASSWORD,
            "host": self.POSTGRES_HOST,
            "port": self.POSTGRES_PORT,
        }
        return data
