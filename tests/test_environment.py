from pathlib import Path


def test_runtime_state_is_redirected_to_isolated_project(isolated_project):
    import cv_agent.config as config
    import cv_agent.documents as documents
    import cv_agent.resource_limits as resource_limits
    import cv_agent.store as store
    import cv_agent.webapp as webapp

    repository_root = Path(__file__).resolve().parents[1]
    assert isolated_project != repository_root
    assert config.ROOT == isolated_project
    assert config.DATA_DIR == isolated_project / "data"
    assert documents.CV_DIR == isolated_project / "cv"
    assert resource_limits.DATA_DIR == isolated_project / "data"
    assert store.DATA_DIR == isolated_project / "data"
    assert webapp.APPLICATIONS_DIR == isolated_project / "applications"

    database = store.Store()
    try:
        assert database.path == isolated_project / "data" / "applications.db"
    finally:
        database.close()
