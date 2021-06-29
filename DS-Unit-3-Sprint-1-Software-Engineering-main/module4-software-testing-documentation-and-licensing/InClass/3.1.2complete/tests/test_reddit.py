import pytest
from reddit import User


def test_user_create():
    mod_status = 9
    username = 'name'
    test_user = User(mod_status, username)
    assert(type(test_user == User))