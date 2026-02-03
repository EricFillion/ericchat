import bleach
import markdown

from ..eric_state import EricUIState
from ..style import EricColours
from ..util import ChatMessage

ALLOWED_TAGS = {
    'p','span','table','thead','tbody','tr','th','td','code','pre',
    'ul','ol','li','br','hr','h1','h2','h3','h4','h5','h6',
    'blockquote','em','strong','caption','tfoot','colgroup','col'
}
ALLOWED_ATTRS = {
    'td': ['align'],
    'th': ['align'],
}

def _render_markdown_to_html(text: str) -> str:
    raw_html = markdown.markdown(text, extensions=['extra', 'sane_lists'])
    clean_html = bleach.clean(
        raw_html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        protocols=[],
        strip=True,
        strip_comments=True
    )
    return clean_html

def _get_item(msg: ChatMessage) -> str:
    if not msg.role:
        return ""

    who_map = {"user": "You", "assistant": "Eric"}

    cls = msg.role
    who = who_map[msg.role]
    html_msg = _render_markdown_to_html(msg.text)

    expanded_html = ""

    tps_value = msg.tps
    tps_chip = ""
    if msg.role == "assistant" and tps_value > 0:
        tps_chip = f'<div class="tps-chip">TPS:<span class="value">{round(tps_value, 1)}</span></div>'

    return f"""
       <div class="row {cls}">
           {tps_chip}
           <div class="who">{who}</div>
           <div class="msg">{html_msg}{expanded_html}</div>
       </div>
    """


def render_html(eric_state: EricUIState):
    items = []
    for msg in eric_state.convo_history:
        item = _get_item(msg)
        if item:
            items.append(item)

    item = _get_item(eric_state.current_marker_stream)
    if item:
        items.append(item)

    transcript = "\n".join(items) or """"""

    return get_html(transcript)

def get_html(transcript):
    message_html = f"""<!doctype html>
    <html lang="en">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      :root {{
        --bg:  {EricColours.DARK_RED};
        --panel: {EricColours.ERIC_DARK_SILVER};
        --accent: {EricColours.ERIC_BLUE};
        --user: {EricColours.ERIC_BLUE};
        --assistant: {EricColours.ERIC_RED};
        --bubble-bg:  {EricColours.LIGHT_RED};
        --bubble-border: {EricColours.ERIC_DARK_SILVER};
      }}

      * {{ box-sizing: border-box; }}
      html {{ scroll-behavior: auto; }}
      body {{ scroll-behavior: auto; }}

      html, body {{
        margin: 0; padding: 0;
        font-family: Roboto, system-ui, sans-serif;
        background: var(--bg);
        color: #111;
      }}

      main {{ padding: 16px; }}

      .timeline {{
        display: grid; gap: 10px;
        overflow-anchor: none;
      }}

      .row {{
        display: grid; gap: 6px;
        background: var(--bubble-bg);
        border: 1px solid var(--bubble-border);
        border-radius: 10px;
        padding: 10px 12px;
        position: relative; /* allow TPS chip to pin inside the box */
      }}
      .row.user {{ border-left: 6px solid var(--user); }}
      .row.assistant {{ border-left: 6px solid var(--assistant); }}

      .tps-chip {{
        position: absolute;
        top: 6px; right: 8px;
        padding: 4px 6px;
        background: #fff;
        color: {EricColours.BLACK};
        border: 1px solid var(--bubble-border);
        border-radius: 999px;
        box-shadow: 0 1px 2px #0002;
        font-size: 10px;
        font-weight: 600;
      }}
      .tps-chip .value {{
        margin-left: 4px;
        color: {EricColours.ERIC_RED};
      }}

      .who {{
        font-weight: 600; font-size: 10px; opacity: .7; color: {EricColours.BLACK};
      }}
      .msg {{
        font-size: 14px; line-height: 1.45; white-space: normal; word-wrap: break-word; color: {EricColours.BLACK};
      }}
      .msg table {{ width: 100%; border-collapse: collapse; margin: 6px 0; }}
      .msg th, .msg td {{ border: 1px solid var(--bubble-border); padding: 6px 8px; text-align: left; }}
      .msg thead th {{ background: #fff8; }}

      footer {{ height: 24px; }}
    </style>
    <body>
      <main>
        <section class="timeline" id="timeline">
          {transcript}
        </section>
      </main>
      <footer id="bottom"></footer>

      <script>

        if ('scrollRestoration' in history) {{
          history.scrollRestoration = 'manual';
        }}
        const SCROLL_KEY = 'eric-chat-scrollY';
        function saveScroll() {{
          sessionStorage.setItem(SCROLL_KEY, String(window.scrollY || 0));
        }}
        window.addEventListener('pagehide', saveScroll);

        function whenLayoutReady() {{
          const fontReady = (document.fonts && document.fonts.ready) ? document.fonts.ready : Promise.resolve();
          const pendingImgs = Array.from(document.images).filter(img => !img.complete);
          const imgReady = Promise.all(pendingImgs.map(img => new Promise(res => {{
            img.addEventListener('load', res, {{ once: true }});
            img.addEventListener('error', res, {{ once: true }});
          }})));
          return Promise.all([fontReady, imgReady]);
        }}

        function robustRestore(y) {{
          window.scrollTo(0, y);
          let attempts = 0;
          function rafKick() {{
            window.scrollTo(0, y);
            if (++attempts < 3) requestAnimationFrame(rafKick);
          }}
          requestAnimationFrame(rafKick);
          let lastHeight = 0;
          const ro = new ResizeObserver(() => {{
            const h = document.documentElement.scrollHeight;
            if (h !== lastHeight) {{
              lastHeight = h;
              window.scrollTo(0, y);
            }}
          }});
          ro.observe(document.documentElement);
          setTimeout(() => ro.disconnect(), 800);
        }}

        document.addEventListener('DOMContentLoaded', () => {{
          const stored = sessionStorage.getItem(SCROLL_KEY);
          if (stored !== null) {{
            const y = parseInt(stored, 10) || 0;
            whenLayoutReady().finally(() => robustRestore(y));
          }} else {{
            document.getElementById('bottom')?.scrollIntoView({{ block: 'end', behavior: 'auto' }});
          }}
        }});
      </script>

    </body>
    </html>
    """
    return message_html
