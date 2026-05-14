"""UI theme stylesheets for the BDG viewer.

Each theme is a dict with:
  - name: display name
  - app_qss: QSS applied to QMainWindow / global controls
  - overlay_qss: QSS applied to the help/preview overlay labels
  - text_color: primary readable text color (used inline in HTML help)
  - muted_color: secondary text color
  - viewer_clear: (r, g, b) background for the GL viewport (0..1)
  - path_label_qss: QSS for the small file/path label

Themes live in utils/ so the main module just imports them.
"""

THEMES = {
    'dark': {
        'name': 'Dark',
        'text_color': '#e6e8ec',
        'muted_color': '#9aa1ab',
        'viewer_clear': (0.10, 0.11, 0.13),
        'app_qss': (
            'QMainWindow{background:#1c1f24;}'
            'QWidget{color:#e6e8ec;}'
            'QPushButton, QToolButton{background:#2a2f37;color:#e6e8ec;border:1px solid #3a414c;'
            'padding:6px 12px;border-radius:4px;}'
            'QPushButton:hover, QToolButton:hover{background:#343b45;}'
            'QPushButton:pressed, QToolButton:pressed{background:#3f4753;}'
            'QToolButton::menu-indicator{image:none;}'
            'QMenu{background:#23272e;color:#e6e8ec;border:1px solid #3a414c;}'
            'QMenu::item:selected{background:#343b45;}'
            'QStatusBar{color:#9aa1ab;background:#1c1f24;}'
        ),
        'overlay_qss': (
            'QLabel{background:rgba(20,22,26,170);color:#e6e8ec;'
            'border:1px solid rgba(58,65,76,160);border-radius:6px;'
            'padding:10px 14px;}'
        ),
        'preview_qss': (
            'QLabel{background:rgba(20,22,26,200);color:#e6e8ec;'
            'border:1px solid rgba(58,65,76,180);border-radius:6px;'
            'padding:6px;}'
        ),
        'path_label_qss': 'color:#9aa1ab;padding:4px 8px;font-size:11px;background:#1c1f24;',
    },
    'light': {
        'name': 'Light',
        'text_color': '#1f2329',
        'muted_color': '#5b6470',
        'viewer_clear': (0.92, 0.93, 0.95),
        'app_qss': (
            'QMainWindow{background:#f3f4f7;}'
            'QWidget{color:#1f2329;}'
            'QPushButton, QToolButton{background:#ffffff;color:#1f2329;border:1px solid #c6cbd2;'
            'padding:6px 12px;border-radius:4px;}'
            'QPushButton:hover, QToolButton:hover{background:#eaecf0;}'
            'QPushButton:pressed, QToolButton:pressed{background:#dde0e6;}'
            'QToolButton::menu-indicator{image:none;}'
            'QMenu{background:#ffffff;color:#1f2329;border:1px solid #c6cbd2;}'
            'QMenu::item:selected{background:#dde7f5;}'
            'QStatusBar{color:#5b6470;background:#f3f4f7;}'
        ),
        'overlay_qss': (
            'QLabel{background:rgba(255,255,255,210);color:#1f2329;'
            'border:1px solid rgba(160,170,185,180);border-radius:6px;'
            'padding:10px 14px;}'
        ),
        'preview_qss': (
            'QLabel{background:rgba(255,255,255,230);color:#1f2329;'
            'border:1px solid rgba(160,170,185,200);border-radius:6px;'
            'padding:6px;}'
        ),
        'path_label_qss': 'color:#5b6470;padding:4px 8px;font-size:11px;background:#f3f4f7;',
    },
    'xp': {
        'name': 'Windows XP',
        'text_color': '#000000',
        'muted_color': '#1d3a8a',
        'viewer_clear': (0.42, 0.62, 0.86),
        'app_qss': (
            'QMainWindow{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,'
            'stop:0 #2a5fc8, stop:0.5 #3a7be0, stop:1 #1d3a8a);}'
            'QWidget{color:#000000;font-family:"Tahoma","MS Sans Serif",sans-serif;}'
            'QPushButton, QToolButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,'
            'stop:0 #fefefe, stop:0.5 #ece9d8, stop:1 #c8c2a6);'
            'color:#000000;border:1px solid #003c74;padding:4px 14px;border-radius:3px;}'
            'QPushButton:hover, QToolButton:hover{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,'
            'stop:0 #ffffe1, stop:0.5 #fff4b0, stop:1 #e6c97a);border:1px solid #b6651a;}'
            'QPushButton:pressed, QToolButton:pressed{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,'
            'stop:0 #c8c2a6, stop:1 #ece9d8);}'
            'QToolButton::menu-indicator{image:none;}'
            'QMenu{background:#ece9d8;color:#000000;border:1px solid #003c74;}'
            'QMenu::item:selected{background:#316ac5;color:#ffffff;}'
            'QStatusBar{color:#ffffff;background:#1d3a8a;}'
        ),
        'overlay_qss': (
            'QLabel{background:rgba(236,233,216,235);color:#000000;'
            'border:1px solid #003c74;border-radius:4px;'
            'padding:10px 14px;}'
        ),
        'preview_qss': (
            'QLabel{background:rgba(236,233,216,245);color:#000000;'
            'border:1px solid #003c74;border-radius:4px;'
            'padding:6px;}'
        ),
        'path_label_qss': (
            'color:#ffffff;padding:4px 8px;font-size:11px;background:#1d3a8a;'
            'font-family:"Tahoma","MS Sans Serif",sans-serif;'
        ),
    },
}


THEMES['win98'] = {
    'name': 'Windows 98',
    'text_color': '#000000',
    'muted_color': '#404040',
    'viewer_clear': (0.0, 0.50, 0.50),
    'app_qss': (
        'QMainWindow{background:#c0c0c0;}'
        'QWidget{color:#000000;font-family:"Tahoma","Segoe UI",sans-serif;font-size:11px;}'
        'QPushButton, QToolButton{background:#c0c0c0;color:#000000;'
        'border-top:1px solid #ffffff;border-left:1px solid #ffffff;'
        'border-right:1px solid #404040;border-bottom:1px solid #404040;'
        'padding:3px 12px;border-radius:0px;}'
        'QPushButton:hover, QToolButton:hover{background:#d4d0c8;}'
        'QPushButton:pressed, QToolButton:pressed{'
        'border-top:1px solid #404040;border-left:1px solid #404040;'
        'border-right:1px solid #ffffff;border-bottom:1px solid #ffffff;}'
        'QToolButton::menu-indicator{image:none;}'
        'QMenu{background:#c0c0c0;color:#000000;border:1px solid #404040;}'
        'QMenu::item:selected{background:#000080;color:#ffffff;}'
        'QStatusBar{color:#000000;background:#c0c0c0;}'
        'QComboBox{background:#ffffff;color:#000000;'
        'border-top:1px solid #404040;border-left:1px solid #404040;'
        'border-right:1px solid #ffffff;border-bottom:1px solid #ffffff;padding:2px 6px;}'
    ),
    'overlay_qss': (
        'QLabel{background:rgba(192,192,192,235);color:#000000;'
        'border-top:2px solid #ffffff;border-left:2px solid #ffffff;'
        'border-right:2px solid #404040;border-bottom:2px solid #404040;'
        'padding:10px 14px;}'
    ),
    'preview_qss': (
        'QLabel{background:rgba(192,192,192,245);color:#000000;'
        'border-top:2px solid #ffffff;border-left:2px solid #ffffff;'
        'border-right:2px solid #404040;border-bottom:2px solid #404040;padding:6px;}'
    ),
    'path_label_qss': (
        'color:#ffffff;padding:4px 8px;font-size:11px;background:#000080;'
        'font-family:"Tahoma","Segoe UI",sans-serif;'
    ),
}

THEMES['solarized'] = {
    'name': 'Solarized Dark',
    'text_color': '#fdf6e3',
    'muted_color': '#93a1a1',
    'viewer_clear': (0.00, 0.17, 0.21),
    'app_qss': (
        'QMainWindow{background:#002b36;}'
        'QWidget{color:#fdf6e3;}'
        'QPushButton, QToolButton{background:#073642;color:#fdf6e3;border:1px solid #586e75;'
        'padding:6px 12px;border-radius:4px;}'
        'QPushButton:hover, QToolButton:hover{background:#0a4250;}'
        'QPushButton:pressed, QToolButton:pressed{background:#114452;}'
        'QToolButton::menu-indicator{image:none;}'
        'QMenu{background:#073642;color:#fdf6e3;border:1px solid #586e75;}'
        'QMenu::item:selected{background:#268bd2;color:#fdf6e3;}'
        'QStatusBar{color:#93a1a1;background:#002b36;}'
        'QComboBox{background:#073642;color:#fdf6e3;border:1px solid #586e75;padding:4px 8px;border-radius:3px;}'
    ),
    'overlay_qss': (
        'QLabel{background:rgba(7,54,66,210);color:#fdf6e3;'
        'border:1px solid #268bd2;border-radius:6px;padding:10px 14px;}'
    ),
    'preview_qss': (
        'QLabel{background:rgba(7,54,66,235);color:#fdf6e3;'
        'border:1px solid #268bd2;border-radius:6px;padding:6px;}'
    ),
    'path_label_qss': 'color:#93a1a1;padding:4px 8px;font-size:11px;background:#002b36;',
}

THEMES['amber'] = {
    'name': 'Amber CRT',
    'text_color': '#ffb000',
    'muted_color': '#a06800',
    'viewer_clear': (0.04, 0.02, 0.0),
    'app_qss': (
        'QMainWindow{background:#0a0703;}'
        'QWidget{color:#ffb000;font-family:"Consolas","Courier New",monospace;}'
        'QPushButton, QToolButton{background:#1a0e00;color:#ffb000;border:1px solid #ffb000;'
        'padding:6px 12px;border-radius:0px;}'
        'QPushButton:hover, QToolButton:hover{background:#3a2200;color:#ffd060;}'
        'QPushButton:pressed, QToolButton:pressed{background:#5a3300;}'
        'QToolButton::menu-indicator{image:none;}'
        'QMenu{background:#0a0703;color:#ffb000;border:1px solid #ffb000;}'
        'QMenu::item:selected{background:#ffb000;color:#0a0703;}'
        'QStatusBar{color:#a06800;background:#0a0703;}'
        'QComboBox{background:#1a0e00;color:#ffb000;border:1px solid #ffb000;padding:4px 8px;}'
    ),
    'overlay_qss': (
        'QLabel{background:rgba(10,7,3,220);color:#ffb000;'
        'border:1px solid #ffb000;border-radius:0px;padding:10px 14px;}'
    ),
    'preview_qss': (
        'QLabel{background:rgba(10,7,3,240);color:#ffb000;'
        'border:1px solid #ffb000;border-radius:0px;padding:6px;}'
    ),
    'path_label_qss': 'color:#a06800;padding:4px 8px;font-size:11px;background:#0a0703;font-family:"Consolas",monospace;',
}

THEMES['matrix'] = {
    'name': 'Matrix',
    'text_color': '#00ff66',
    'muted_color': '#00993d',
    'viewer_clear': (0.0, 0.02, 0.0),
    'app_qss': (
        'QMainWindow{background:#000000;}'
        'QWidget{color:#00ff66;font-family:"Consolas","Courier New",monospace;}'
        'QPushButton, QToolButton{background:#001a08;color:#00ff66;border:1px solid #00ff66;'
        'padding:6px 12px;border-radius:2px;}'
        'QPushButton:hover, QToolButton:hover{background:#003315;}'
        'QPushButton:pressed, QToolButton:pressed{background:#005c25;}'
        'QToolButton::menu-indicator{image:none;}'
        'QMenu{background:#000000;color:#00ff66;border:1px solid #00ff66;}'
        'QMenu::item:selected{background:#00ff66;color:#000000;}'
        'QStatusBar{color:#00993d;background:#000000;}'
        'QComboBox{background:#001a08;color:#00ff66;border:1px solid #00ff66;padding:4px 8px;}'
    ),
    'overlay_qss': (
        'QLabel{background:rgba(0,0,0,220);color:#00ff66;'
        'border:1px solid #00ff66;border-radius:2px;padding:10px 14px;}'
    ),
    'preview_qss': (
        'QLabel{background:rgba(0,0,0,240);color:#00ff66;'
        'border:1px solid #00ff66;border-radius:2px;padding:6px;}'
    ),
    'path_label_qss': 'color:#00993d;padding:4px 8px;font-size:11px;background:#000000;font-family:"Consolas",monospace;',
}


def get_theme(key: str) -> dict:
    return THEMES.get(key, THEMES['dark'])
