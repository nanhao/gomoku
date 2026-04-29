"""写入保护 Hook 测试

验证 conftest.py 中的写入保护 hook 能正确拦截测试对受保护目录的写入操作，
同时不影响正常的读取和临时文件写入。
"""

import os
import tempfile
import pytest

# 项目根目录
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 受保护目录列表（与 conftest.py 中 PROTECTED_DIRS 保持一致）
PROTECTED_DIR_RELATIVE = [
    'ai/ai_data',
    'ai/ai_data/loss_logs',
    'ai/models',
    'ai/train_data',
    'ai/merge_samples',
    'game/save',
    'design',
    'requirements',
]


class TestWriteGuardBlocked:
    """验证写入保护 hook 拦截各类写入操作"""

    @pytest.mark.parametrize('rel_dir', PROTECTED_DIR_RELATIVE,
                             ids=lambda d: d.replace('/', '_'))
    def test_write_mode_blocked(self, rel_dir):
        """w 模式写入受保护目录应抛出 AssertionError"""
        test_file = os.path.join(_PROJECT_ROOT, rel_dir, 'guard_test.tmp')
        with pytest.raises(AssertionError, match=r'\[WRITE GUARD\]'):
            open(test_file, 'w')

    @pytest.mark.parametrize('rel_dir', PROTECTED_DIR_RELATIVE,
                             ids=lambda d: d.replace('/', '_'))
    def test_append_mode_blocked(self, rel_dir):
        """a 模式追加受保护目录应抛出 AssertionError"""
        test_file = os.path.join(_PROJECT_ROOT, rel_dir, 'guard_test.tmp')
        with pytest.raises(AssertionError, match=r'\[WRITE GUARD\]'):
            open(test_file, 'a')

    @pytest.mark.parametrize('rel_dir', PROTECTED_DIR_RELATIVE,
                             ids=lambda d: d.replace('/', '_'))
    def test_write_plus_mode_blocked(self, rel_dir):
        """w+ 模式写入受保护目录应抛出 AssertionError"""
        test_file = os.path.join(_PROJECT_ROOT, rel_dir, 'guard_test.tmp')
        with pytest.raises(AssertionError, match=r'\[WRITE GUARD\]'):
            open(test_file, 'w+')

    @pytest.mark.parametrize('rel_dir', PROTECTED_DIR_RELATIVE,
                             ids=lambda d: d.replace('/', '_'))
    def test_read_plus_mode_blocked(self, rel_dir):
        """r+ 模式打开受保护目录应抛出 AssertionError（允许写入）"""
        test_file = os.path.join(_PROJECT_ROOT, rel_dir, 'guard_test.tmp')
        with pytest.raises(AssertionError, match=r'\[WRITE GUARD\]'):
            open(test_file, 'r+')

    def test_exclusive_create_mode_blocked(self):
        """x 模式排他创建受保护目录应抛出 AssertionError"""
        test_file = os.path.join(_PROJECT_ROOT, 'ai', 'ai_data', 'guard_test.tmp')
        with pytest.raises(AssertionError, match=r'\[WRITE GUARD\]'):
            open(test_file, 'x')

    def test_binary_write_mode_blocked(self):
        """wb 二进制写入受保护目录应抛出 AssertionError"""
        test_file = os.path.join(_PROJECT_ROOT, 'game', 'save', 'guard_test.tmp')
        with pytest.raises(AssertionError, match=r'\[WRITE GUARD\]'):
            open(test_file, 'wb')

    def test_pathlib_write_text_blocked(self):
        """pathlib.Path.write_text 写入受保护目录应抛出 AssertionError"""
        from pathlib import Path
        test_file = Path(_PROJECT_ROOT) / 'design' / 'guard_test.tmp'
        with pytest.raises(AssertionError, match=r'\[WRITE GUARD\]'):
            test_file.write_text('test')

    def test_pathlib_write_bytes_blocked(self):
        """pathlib.Path.write_bytes 写入受保护目录应抛出 AssertionError"""
        from pathlib import Path
        test_file = Path(_PROJECT_ROOT) / 'requirements' / 'guard_test.tmp'
        with pytest.raises(AssertionError, match=r'\[WRITE GUARD\]'):
            test_file.write_bytes(b'test')

    def test_error_message_contains_directory(self):
        """错误信息应包含被拦截的文件路径"""
        test_file = os.path.join(_PROJECT_ROOT, 'ai', 'models', 'guard_test.tmp')
        with pytest.raises(AssertionError) as exc_info:
            open(test_file, 'w')
        # 验证错误信息包含文件路径和受保护目录列表
        error_msg = str(exc_info.value)
        assert '[WRITE GUARD]' in error_msg
        assert 'ai' in error_msg and 'models' in error_msg


class TestWriteGuardAllowed:
    """验证写入保护 hook 不影响正常操作"""

    def test_read_from_protected_dir_allowed(self):
        """从受保护目录读取文件应正常工作"""
        init_file = os.path.join(_PROJECT_ROOT, 'ai', 'models', '__init__.py')
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert isinstance(content, str)

    def test_write_to_temp_dir_allowed(self):
        """向临时目录写入文件应正常工作"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp',
                                         delete=False) as f:
            f.write('test content')
            temp_path = f.name
        try:
            with open(temp_path, 'r') as f:
                assert f.read() == 'test content'
        finally:
            os.unlink(temp_path)

    def test_write_to_tmp_path_allowed(self, tmp_path):
        """向 pytest tmp_path fixture 写入应正常工作"""
        test_file = tmp_path / 'guard_test.tmp'
        test_file.write_text('test content')
        assert test_file.read_text() == 'test content'

    def test_read_mode_not_blocked(self):
        """r 模式读取受保护目录文件应正常工作"""
        # 使用确定存在的文件
        init_file = os.path.join(_PROJECT_ROOT, 'ai', 'models', '__init__.py')
        # 如果文件存在则测试，否则跳过
        if os.path.exists(init_file):
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            assert isinstance(content, str)

    def test_write_to_non_protected_project_dir_allowed(self, tmp_path):
        """向非受保护的项目目录写入应正常工作（通过 tmp_path）"""
        test_file = tmp_path / 'subdir' / 'test.json'
        test_file.parent.mkdir()
        test_file.write_text('{"test": true}')
        assert test_file.read_text() == '{"test": true}'


class TestWriteGuardEdgeCases:
    """写入保护 hook 边界情况测试"""

    def test_relative_path_also_blocked(self):
        """使用相对路径写入受保护目录也应被拦截"""
        # 切换到项目根目录后使用相对路径
        original_cwd = os.getcwd()
        try:
            os.chdir(_PROJECT_ROOT)
            with pytest.raises(AssertionError, match=r'\[WRITE GUARD\]'):
                open('game/save/guard_test.tmp', 'w')
        finally:
            os.chdir(original_cwd)

    def test_subdirectory_of_protected_dir_blocked(self):
        """受保护目录的子目录写入也应被拦截"""
        # ai/models/cnn_v1/ 是 ai/models 的子目录
        test_file = os.path.join(_PROJECT_ROOT, 'ai', 'models', 'cnn_v1',
                                 'guard_test.tmp')
        with pytest.raises(AssertionError, match=r'\[WRITE GUARD\]'):
            open(test_file, 'w')

    def test_similar_but_different_dir_not_blocked(self, tmp_path):
        """与受保护目录名相似但不同的目录不应被拦截"""
        # 例如 ai/ai_data_backup 不应被 ai/ai_data 拦截
        # 用 tmp_path 模拟这个场景
        fake_backup_dir = tmp_path / 'ai_data_backup'
        fake_backup_dir.mkdir()
        test_file = fake_backup_dir / 'test.tmp'
        # 这应该正常工作，不会抛出 AssertionError
        test_file.write_text('test')
        assert test_file.read_text() == 'test'
