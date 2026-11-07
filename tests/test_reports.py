from albert.reports import Report


def test_same_name_reports_in_one_second_get_their_own_folders(tmp_path):
    a, b, c = Report(tmp_path, "x"), Report(tmp_path, "x"), Report(tmp_path, "x")
    assert len({a.dir, b.dir, c.dir}) == 3
    assert all(p.is_dir() for p in (a.dir, b.dir, c.dir))
