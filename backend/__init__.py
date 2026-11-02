"""Backend package: core engine, database, services and HTTP API.

Layout
------
core/       Pure reasoning engine (no I/O framework, no web code)
db/         SQLAlchemy models, session handling and repositories
services/   Application services that wire core + db together
api/        FastAPI application, routes and request/response schemas
"""

__version__ = "2.0.0"
