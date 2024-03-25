import psycopg2.pool

from psycopg2.extras import RealDictCursor

DATABASE_HOST = 'localhost'
DATABASE_USERNAME = 'postgres'
DATABASE_PASSWORD = 'pje'
DATABASE_PORT = '5432'
DATABASE_NAME = 'llm_experiments'
DATABASE_POOL_MIN = 1
DATABASE_POOL_MAX = 10

class SingletonMeta(type):
    """ Singleton metaclass """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """ Call method """

        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]


class Database(metaclass=SingletonMeta):
    """ Database class to handle the connection pool """

    def __init__(self):
        """ Database constructor """

        self.host = DATABASE_HOST
        self.username = DATABASE_USERNAME
        self.password = DATABASE_PASSWORD
        self.port = DATABASE_PORT or 5432
        self.dbname = DATABASE_NAME
        self.minconn = DATABASE_POOL_MIN or 1
        self.maxconn = DATABASE_POOL_MAX or 10

        self.pool = None

        self.connect()

    def connect(self):
        """ Connect to the database """

        if self.pool is None:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                self.minconn,
                self.maxconn,
                host=self.host,
                user=self.username,
                password=self.password,
                port=self.port,
                database=self.dbname,
            )

    def query(self, query: str, record: tuple = None):
        """ Execute a query """

        conn = None
        cur = None

        try:
            conn = self.pool.getconn()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute(query, record)
            conn.commit()

            if cur.description:
                result = cur.fetchall()
            else:
                result = None

            return result

        finally:
            if cur:
                cur.close()
            if conn:
                self.pool.putconn(conn)

    def close(self):
        """ Close the connection pool """

        self.pool.closeall()
