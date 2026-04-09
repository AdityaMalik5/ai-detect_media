"""Tests for the shared utils module."""

from utils import allowed_file, get_filename_only


class TestGetFilenameOnly:
    def test_simple_filename(self):
        assert get_filename_only("video.mp4") == "video"

    def test_full_path(self):
        assert get_filename_only("/home/user/data/clip.avi") == "clip"

    def test_multiple_dots(self):
        assert get_filename_only("my.video.file.mp4") == "my"

    def test_no_extension(self):
        assert get_filename_only("README") == "README"

    def test_nested_path(self):
        assert get_filename_only("a/b/c/frame-001.png") == "frame-001"


class TestAllowedFile:
    def test_png(self):
        assert allowed_file("face.png") is True

    def test_jpg(self):
        assert allowed_file("photo.jpg") is True

    def test_jpeg(self):
        assert allowed_file("photo.JPEG") is True

    def test_bmp(self):
        assert allowed_file("img.bmp") is True

    def test_webp(self):
        assert allowed_file("img.webp") is True

    def test_reject_mp4(self):
        assert allowed_file("video.mp4") is False

    def test_reject_txt(self):
        assert allowed_file("notes.txt") is False

    def test_reject_exe(self):
        assert allowed_file("malware.exe") is False

    def test_reject_no_extension(self):
        assert allowed_file("noext") is False

    def test_reject_empty(self):
        assert allowed_file("") is False
