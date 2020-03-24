# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['Machine_Learning_Tool_Unification_Service.py'],
             pathex=['C:\\Users\\Z003XE7X\\Documents\\Unification Service Machine Learning Application\\ML Application Unification Service\\Machine_Learning_Tool'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='Machine_Learning_Tool_Unification_Service',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Machine_Learning_Tool_Unification_Service')
