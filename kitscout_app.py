"""
KitScout — AI Agent สำหรับนักสะสมเสื้อฟุตบอล
Streamlit App with Google Gemini + Image Upload
"""

import streamlit as st
import google.generativeai as genai
import json
from PIL import Image
import io

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="KitScout",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# API Key — จาก .streamlit/secrets.toml
# ──────────────────────────────────────────────
def get_api_key() -> str:
    # 1. ลองจาก Streamlit secrets ก่อน
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    # 2. fallback: ให้ user พิมพ์ใน sidebar
    return ""


# ──────────────────────────────────────────────
# Tool Definitions
# ──────────────────────────────────────────────
KITSCOUT_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "search_kits",
                "description": "ค้นหาเสื้อฟุตบอลจากหลายแพลตฟอร์ม เช่น Facebook, Shopee, Lazada, Carousell, eBay",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "ชื่อเสื้อที่ต้องการค้นหา"
                        },
                        "size": {
                            "type": "string",
                            "description": "ไซส์ เช่น S, M, L, XL (optional)"
                        },
                        "platform": {
                            "type": "string",
                            "description": "แพลตฟอร์ม หรือ 'all'"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "check_authenticity",
                "description": "วิเคราะห์ความแท้ของเสื้อฟุตบอลจากรายละเอียดและ/หรือรูปภาพที่ผู้ใช้ให้มา",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kit_name": {
                            "type": "string",
                            "description": "ชื่อเสื้อที่ต้องการตรวจสอบ"
                        },
                        "details": {
                            "type": "string",
                            "description": "รายละเอียด เช่น product code, tag, stitching"
                        },
                        "has_image": {
                            "type": "boolean",
                            "description": "ผู้ใช้แนบรูปมาด้วยหรือไม่"
                        }
                    },
                    "required": ["kit_name", "details"]
                }
            },
            {
                "name": "analyze_market_price",
                "description": "วิเคราะห์ราคาตลาดและความคุ้มค่าของเสื้อฟุตบอล",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kit_name": {
                            "type": "string",
                            "description": "ชื่อเสื้อ"
                        },
                        "offered_price": {
                            "type": "number",
                            "description": "ราคาที่เสนอ (บาท)"
                        },
                        "condition": {
                            "type": "string",
                            "description": "สภาพ: BNWT, BNWOT, Excellent, Good, Used"
                        }
                    },
                    "required": ["kit_name"]
                }
            },
            {
                "name": "manage_wishlist",
                "description": "จัดการ Wishlist — เพิ่ม ลบ ดูรายการ ล้างทั้งหมด",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "remove", "list", "clear"],
                            "description": "add / remove / list / clear"
                        },
                        "kit_name": {
                            "type": "string",
                            "description": "ชื่อเสื้อ (ต้องการเมื่อ add/remove)"
                        },
                        "target_price": {
                            "type": "number",
                            "description": "ราคาเป้าหมาย (บาท)"
                        },
                        "size": {
                            "type": "string",
                            "description": "ไซส์ที่ต้องการ"
                        }
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "get_kit_knowledge",
                "description": "ให้ข้อมูลเชิงลึกเกี่ยวกับ Football Kit ประเภท ประวัติ ศัพท์วงการ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "หัวข้อ เช่น Replica vs Authentic, Player Issue, Match Worn, BNWT"
                        }
                    },
                    "required": ["topic"]
                }
            }
        ]
    }
]

# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
คุณคือ "KitScout" — AI Agent สำหรับนักสะสมเสื้อฟุตบอล (Football Kit Collector)

ภารกิจหลัก:
1. ช่วยหาเสื้อที่ต้องการให้เร็วที่สุด
2. ลดความเสี่ยงจากการซื้อของปลอม
3. ช่วยประเมินความคุ้มค่าของราคา
4. ให้ความรู้เกี่ยวกับ Football Kit อย่างถูกต้อง

## Core Principles
ลำดับความสำคัญ: ความถูกต้อง > ความรวดเร็ว > ความครบถ้วน
- หากไม่มีข้อมูลเพียงพอ: บอกตรงๆ และขอข้อมูลเพิ่ม
- ห้ามเดาหรือสร้างข้อมูลขึ้นเอง

## Data Source Rules
ห้ามสร้างข้อมูลต่อไปนี้ขึ้นเอง: Listing, ราคาขาย, ผู้ขาย, จำนวนสต็อก, ลิงก์, Product Code เฉพาะ
หากไม่สามารถเข้าถึง Real-time: บอกตรงๆ และแนะนำแหล่งค้นหาแทน

## เมื่อมีรูปภาพ
วิเคราะห์รูปภาพอย่างละเอียด ดู:
- Badge/Crest: ความละเอียดของการปัก ตำแหน่ง
- Font ชื่อ-เบอร์: ความหนา ระยะห่าง สัดส่วน
- Fabric Tag: ข้อความ RN# GLN# country of manufacture
- Sponsor logo: placement, proportions, stitching
- วัสดุและ texture โดยรวม
- Hologram หรือ authentication label

## Authenticity Check Format
สรุป: น่าจะแท้มาก / มีแนวโน้มแท้ / น่าสงสัย / เสี่ยงปลอม
คะแนนความเชื่อมั่น: X/100
เหตุผลสนับสนุน:
- ...
จุดที่ต้องระวัง:
- ...
ข้อมูลที่ยังขาด:
- ...
ห้ามยืนยันว่าแท้ 100% เด็ดขาด

## Confidence Scale
90-100: น่าจะแท้มาก | 70-89: มีแนวโน้มแท้ | 40-69: น่าสงสัย | 0-39: เสี่ยงปลอม

## Market Price Format
สภาพ: [BNWT/BNWOT/Used]
ช่วงราคาตลาดโดยประมาณ: XXX - XXX บาท
การประเมิน: ✓ ต่ำกว่าตลาด / ≈ ใกล้เคียงตลาด / ⚠ สูงกว่าตลาด
เหตุผล: ...

## Response Style
- ตอบภาษาไทยเป็นหลัก
- ใช้ศัพท์วงการได้: Drop, Deadstock, Player Issue, Match Worn, BNWT, BNWOT, Name Set, Patch, Replica, Authentic
- ทุกคำตอบ: สรุปสั้นก่อน → รายละเอียด → คำแนะนำต่อไป
- ถามกลับได้ครั้งละ 1 คำถามเท่านั้น

## Refusal Rules
- นอกขอบเขต: "เรื่องนี้อยู่นอกความเชี่ยวชาญของ KitScout"
- ของปลอม: "ผมไม่สามารถแนะนำหรือสนับสนุนการซื้อขายสินค้าปลอมได้"
"""

# ──────────────────────────────────────────────
# Tool Handlers
# ──────────────────────────────────────────────
def execute_search_kits(query: str, size: str = None, platform: str = "all") -> dict:
    search_term = f"{query} {size}".strip() if size else query
    platforms = {
        "facebook":  f"https://www.facebook.com/marketplace/search/?query={query.replace(' ', '+')}",
        "shopee":    f"https://shopee.co.th/search?keyword={query.replace(' ', '+')}",
        "lazada":    f"https://www.lazada.co.th/catalog/?q={query.replace(' ', '+')}",
        "carousell": f"https://www.carousell.com/search/{query.replace(' ', '%20')}",
        "ebay":      f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}&_sop=12",
    }
    links = platforms if platform == "all" else {platform: platforms.get(platform, "")}
    return {
        "status": "no_realtime_access",
        "query": search_term,
        "note": "KitScout ยังไม่สามารถดึงข้อมูล Real-time listing ได้โดยอัตโนมัติ",
        "manual_search_links": links,
        "facebook_groups": ["Football Shirt Thailand", "เสื้อฟุตบอล มือสอง ของแท้", "Kit Collector Thailand"],
        "tip": f"ค้น '{search_term}' ใน Facebook Groups นักสะสม มักมีของหายากกว่า marketplace"
    }


def execute_check_authenticity(kit_name: str, details: str, has_image: bool = False) -> dict:
    return {
        "kit_name": kit_name,
        "details_provided": details,
        "has_image": has_image,
        "analysis_checklist": [
            "Product Code format ตรงกับ batch ปีนั้น",
            "Font ชื่อ-เบอร์: ขนาด ความหนา ระยะห่าง",
            "Badge/Crest: stitching ความละเอียด ตำแหน่ง",
            "Fabric Care Tag: RN# GLN# country",
            "Sponsor logo placement และ proportions",
            "Hologram/authentication label",
            "ราคาที่ขายเหมาะสมกับของแท้หรือไม่"
        ]
    }


def execute_analyze_market_price(kit_name: str, offered_price: float = None, condition: str = None) -> dict:
    return {
        "kit_name": kit_name,
        "offered_price": offered_price,
        "condition": condition,
        "status": "knowledge_based_only",
        "note": "ราคาอ้างอิงจาก knowledge base — ตรวจสอบ Real-time ที่ eBay Sold Listings",
        "reference_sources": [
            "eBay Sold Listings: กรอง 'Sold Items'",
            "Carousell Thailand",
            "Facebook Groups นักสะสม"
        ]
    }


def execute_manage_wishlist(action: str, kit_name: str = None,
                             target_price: float = None, size: str = None) -> dict:
    wl = st.session_state.wishlist
    if action == "add":
        if not kit_name:
            return {"status": "error", "message": "ต้องระบุ kit_name"}
        if any(w["kit"].lower() == kit_name.lower() for w in wl):
            return {"status": "already_exists", "total": len(wl)}
        item = {"kit": kit_name, "size": size, "target_price_thb": target_price}
        wl.append(item)
        return {"status": "added", "item": item, "total_items": len(wl)}
    elif action == "remove":
        if not kit_name:
            return {"status": "error", "message": "ต้องระบุ kit_name"}
        before = len(wl)
        st.session_state.wishlist = [w for w in wl if w["kit"].lower() != kit_name.lower()]
        return {"status": "removed", "count_removed": before - len(st.session_state.wishlist)}
    elif action == "list":
        return {"wishlist": wl, "total": len(wl)}
    elif action == "clear":
        count = len(wl)
        st.session_state.wishlist = []
        return {"status": "cleared", "items_removed": count}
    return {"status": "error", "message": f"action ไม่รู้จัก: {action}"}


def execute_get_kit_knowledge(topic: str) -> dict:
    return {
        "topic": topic,
        "instruction": "อธิบายหัวข้อนี้อย่างละเอียดในฐานะผู้เชี่ยวชาญ Football Kit"
    }


_TOOL_HANDLERS = {
    "search_kits":          execute_search_kits,
    "check_authenticity":   execute_check_authenticity,
    "analyze_market_price": execute_analyze_market_price,
    "manage_wishlist":      execute_manage_wishlist,
    "get_kit_knowledge":    execute_get_kit_knowledge,
}


def dispatch_tool(name: str, args: dict) -> str:
    handler = _TOOL_HANDLERS.get(name)
    result = handler(**args) if handler else {"error": f"Unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Gemini Agent Core
# ──────────────────────────────────────────────

# Models ที่รองรับ — เรียงจากแนะนำไปถึง fallback
AVAILABLE_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
]


def get_gemini_model(api_key: str, model_name: str = "gemini-1.5-flash"):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
        tools=KITSCOUT_TOOLS,
    )


def _image_to_part(pil_image: Image.Image) -> dict:
    """แปลง PIL Image เป็น inline_data part ที่ Gemini SDK รับได้แน่นอน"""
    buf = io.BytesIO()
    fmt = pil_image.format or "JPEG"
    if fmt not in ("JPEG", "PNG", "WEBP"):
        fmt = "JPEG"
    pil_image.save(buf, format=fmt)
    return {
        "inline_data": {
            "mime_type": f"image/{fmt.lower()}",
            "data": buf.getvalue(),
        }
    }


def run_agent(model, chat, user_text: str, uploaded_image=None) -> str:
    """Send message (+ optional image) through the agentic tool-use loop."""

    # Build content parts — ส่งเป็น list[str | dict]
    content_parts = []
    if uploaded_image is not None:
        content_parts.append(_image_to_part(uploaded_image))
    if user_text:
        content_parts.append(user_text)

    try:
        response = chat.send_message(content_parts)
    except Exception as e:
        err = str(e)
        if "404" in err or "not found" in err.lower():
            return (
                "⚠️ ไม่พบ model ที่เลือก กรุณาเปลี่ยน model ใน sidebar\n\n"
                f"รายละเอียด: `{err[:300]}`"
            )
        raise

    for _ in range(5):
        has_fc = any(
            hasattr(p, "function_call") and p.function_call
            for p in response.parts
        )
        if not has_fc:
            break

        tool_results = []
        for part in response.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args = dict(fc.args)
                result_str = dispatch_tool(fc.name, args)
                tool_results.append({
                    "function_response": {
                        "name": fc.name,
                        "response": {"result": result_str}
                    }
                })
        if not tool_results:
            break
        response = chat.send_message(tool_results)

    texts = [p.text for p in response.parts if hasattr(p, "text") and p.text]
    return "\n".join(texts) if texts else "ขออภัย ไม่สามารถสร้างคำตอบได้ครับ"


# ──────────────────────────────────────────────
# Session State Init
# ──────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "wishlist": [],
        "chat": None,
        "model": None,
        "api_key_ok": False,
        "_last_cfg": "",
        "prefill": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔍 KitScout")
        st.markdown("*เรดาร์เสื้อบอลส่วนตัวของคุณ*")
        st.divider()

        # API Key section
        st.markdown("### ⚙️ API Key")
        stored_key = get_api_key()

        if stored_key and stored_key != "your-gemini-api-key-here":
            st.success("API Key โหลดจาก secrets แล้ว ✅")
            api_key = stored_key
        else:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIza...",
                help="รับ Key ฟรีที่ aistudio.google.com"
            )

        # Model selector
        selected_model = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS,
            index=0,
            help="gemini-1.5-flash = เร็วและฟรีโควต้าสูง"
        )

        # เชื่อมต่อใหม่เมื่อ key หรือ model เปลี่ยน
        current_cfg = f"{api_key}|{selected_model}"
        if api_key and st.session_state.get("_last_cfg") != current_cfg:
            try:
                with st.spinner("กำลังเชื่อมต่อ..."):
                    model = get_gemini_model(api_key, selected_model)
                    # ทดสอบด้วย request เล็กๆ (ไม่มี tools เพื่อความเร็ว)
                    test_model = genai.GenerativeModel(model_name=selected_model)
                    test_model.generate_content("hi")
                st.session_state.model = model
                st.session_state.chat = model.start_chat(history=[])
                st.session_state.api_key_ok = True
                st.session_state["_last_cfg"] = current_cfg
                st.success(f"เชื่อมต่อสำเร็จ ✅ ({selected_model})")
            except Exception as e:
                st.session_state.api_key_ok = False
                err_msg = str(e)
                if "404" in err_msg or "NotFound" in err_msg or "not found" in err_msg.lower():
                    st.error(f"ไม่พบ model '{selected_model}' — ลองเปลี่ยน model ด้านบน")
                elif "API_KEY" in err_msg or "invalid" in err_msg.lower():
                    st.error("API Key ไม่ถูกต้อง กรุณาตรวจสอบ")
                else:
                    st.error(f"เชื่อมต่อไม่ได้: {err_msg[:200]}")
        elif st.session_state.api_key_ok:
            st.success(f"เชื่อมต่อแล้ว ✅ ({selected_model})")

        st.divider()

        # Wishlist panel
        st.markdown("### 📋 Wishlist")
        wl = st.session_state.get("wishlist", [])
        if wl:
            for i, item in enumerate(wl):
                price_str = f"฿{item['target_price_thb']:,.0f}" if item.get("target_price_thb") else ""
                size_str  = item.get("size") or ""
                label = item["kit"]
                if size_str:
                    label += f" ({size_str})"
                if price_str:
                    label += f" — {price_str}"
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"**{i+1}.** {label}")
                if col2.button("✕", key=f"del_{i}"):
                    st.session_state.wishlist.pop(i)
                    st.rerun()
        else:
            st.markdown("*ยังไม่มีรายการ*")

        if wl:
            if st.button("🗑️ ล้าง Wishlist ทั้งหมด", use_container_width=True):
                st.session_state.wishlist = []
                st.rerun()

        st.divider()

        # Quick examples
        st.markdown("### 💬 ตัวอย่างคำถาม")
        examples = [
            "หา Arsenal Away 2002/03 ไซส์ L",
            "Replica กับ Authentic ต่างกันยังไง",
            "เพิ่ม Liverpool Istanbul 2005 ใน wishlist",
            "เช็คความแท้จากรูปที่แนบมา",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
                st.session_state["prefill"] = ex

        st.divider()
        if st.button("🔄 เริ่มการสนทนาใหม่", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.model:
                st.session_state.chat = st.session_state.model.start_chat(history=[])
            st.rerun()


# ──────────────────────────────────────────────
# Persona & JTBD Landing Page
# ──────────────────────────────────────────────
def _render_persona_page():
    st.divider()

    # ── Persona Card ──────────────────────────
    st.markdown("### 👤 ออกแบบมาสำหรับใคร?")

    with st.container(border=True):
        col_ctx, col_goal = st.columns(2)

        with col_ctx:
            st.markdown("**📍 Context**")
            st.markdown(
                "นักสะสมในกลุ่ม Resell ไทยบน Facebook เช่น "
                "**Kit Collector Thailand** และ **เสื้อฟุตบอล มือสอง ของแท้** "
                "ที่ใช้ Shopee · Lazada · Carousell · eBay เป็นประจำ "
                "และคุ้นเคยกับตลาดเสื้อบอลมือสอง"
            )

        with col_goal:
            st.markdown("**🎯 Goal**")
            st.markdown(
                "ตามล่าเสื้อ **Limited Edition · Vintage · Player Issue** "
                "ของแท้ในราคาคุ้มค่า เพื่อเพิ่มชิ้นไม่ซ้ำใครในคอลเลกชัน"
            )

        st.divider()
        col_beh, col_pain = st.columns(2)

        with col_beh:
            st.markdown("**🧠 Behaviour**")
            st.markdown(
                "- สะสมเสื้อหายากและเสื้อคลาสสิกอย่างจริงจัง\n"
                "- วางแผนซื้อล่วงหน้า ตั้งราคาเป้าหมายและไซส์ไว้ใน Wishlist\n"
                "- รอจนกว่าจะเจอราคาที่ใช่"
            )

        with col_pain:
            st.markdown("**😤 Frustration**")
            st.markdown(
                "- ของกระจายหลายแพลตฟอร์ม เจอของปลอม\n"
                "- ตัดสินใจช้า → พลาดการซื้อ\n"
                "- ราคาตลาดไม่แน่นอน ไม่รู้ว่าแพงหรือถูก"
            )

    st.markdown("")

    # ── JTBD Cards ────────────────────────────
    st.markdown("### 🎯 Jobs-to-be-Done — KitScout ช่วยได้อะไรบ้าง?")

    jtbd_list = [
        {
            "icon": "🔎",
            "num": "1",
            "title": "ค้นหาข้ามแพลตฟอร์ม",
            "when": "อยากหาเสื้อชิ้นหนึ่ง แต่ต้องเปิดหลายแท็บพร้อมกัน",
            "want": "ค้นจากที่เดียว ระบุชื่อ ไซส์ แพลตฟอร์มได้เลย",
            "so":   "เห็นผลทุกแหล่งพร้อมกัน ตัดสินใจเร็วขึ้น",
            "example": "หา Arsenal Away 2002/03 ไซส์ L ทุกแพลตฟอร์ม",
        },
        {
            "icon": "🛡️",
            "num": "2",
            "title": "ตรวจสอบความแท้",
            "when": "เจอเสื้อที่ชอบ แต่ไม่แน่ใจว่าของแท้ มีทั้งรูปและรายละเอียด",
            "want": "วิเคราะห์ Badge · Font · Fabric Tag · Hologram ได้ทันที",
            "so":   "ตัดสินใจซื้อได้มั่นใจ ไม่เสี่ยงเสียเงินกับของปลอม",
            "example": "แนบรูปเสื้อ + พิมพ์ 'เช็คความแท้ Man Utd 1999 Home'",
        },
        {
            "icon": "💰",
            "num": "3",
            "title": "วิเคราะห์ราคาตลาด",
            "when": "คนขายบอกราคา แต่ไม่รู้ว่าแพงหรือถูกเมื่อเทียบตลาด",
            "want": "รู้ช่วงราคา BNWT / BNWOT / Used แยกตามสภาพ พร้อมแหล่งอ้างอิง",
            "so":   "ต่อราคาได้มีข้อมูล หรือรู้ว่าควรหนีจากดีลนี้",
            "example": "เขาขาย Liverpool 2005 Final BNWT ราคา 8,000 บาท แพงไปไหม?",
        },
        {
            "icon": "📋",
            "num": "4",
            "title": "จัดการ Wishlist",
            "when": "เจอเสื้อที่อยากได้แต่ราคายังไม่ใช่ หรือยังไม่มีไซส์",
            "want": "บันทึกไว้พร้อมราคาเป้าหมายและไซส์ รอจนถึงเงื่อนไข",
            "so":   "ไม่พลาดเมื่อของหายากปรากฏในราคาที่รับได้",
            "example": "เพิ่ม Zidane 98 World Cup Final ไว้ใน Wishlist ราคาเป้าหมาย 12,000 บาท",
        },
        {
            "icon": "📚",
            "num": "5",
            "title": "ความรู้เชิงลึก Football Kit",
            "when": "สับสนเรื่อง Replica vs Authentic vs Player Issue หรือไม่รู้ว่า BNWT คืออะไร",
            "want": "คำอธิบายจากผู้เชี่ยวชาญ ไม่ต้องออกจาก App ไปค้น Google",
            "so":   "ตัดสินใจซื้อได้ถูกต้อง และพูดคุยในชุมชนได้อย่างมีความรู้",
            "example": "Player Issue กับ Match Worn ต่างกันยังไง ราคาห่างกันมากไหม?",
        },
    ]

    for j in jtbd_list:
        with st.expander(f"{j['icon']} JTBD {j['num']}: {j['title']}", expanded=False):
            cols = st.columns([1, 1, 1])
            with cols[0]:
                st.markdown("**🕐 When**")
                st.caption(j["when"])
            with cols[1]:
                st.markdown("**💡 I want to**")
                st.caption(j["want"])
            with cols[2]:
                st.markdown("**✅ So I can**")
                st.caption(j["so"])
            st.markdown(
                f"<div style='margin-top:8px; padding:8px 12px; "
                f"background:#f0f4ff; border-left:3px solid #4f8ef7; "
                f"border-radius:4px; font-size:0.85rem; color:#333;'>"
                f"💬 <b>ลองพิมพ์:</b> {j['example']}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")


# ──────────────────────────────────────────────
# Main Chat UI
# ──────────────────────────────────────────────
def render_chat():
    st.markdown("## 🔍 KitScout — เรดาร์เสื้อบอลส่วนตัวของคุณ")

    # Persona & JTBD box — แสดงเสมอ พับ/ขยายได้
    with st.expander("👤 เกี่ยวกับ KitScout — Persona & Jobs-to-be-Done", expanded=False):
        _render_persona_page()

    if not st.session_state.api_key_ok:
        st.info("👈 ใส่ Gemini API Key ใน sidebar เพื่อเริ่มใช้งาน")
        return

    st.divider()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("image"):
                st.image(msg["image"], caption="รูปที่แนบมา", use_container_width=True)
            st.markdown(msg["content"])

    # Image uploader (above input)
    uploaded_file = st.file_uploader(
        "📎 แนบรูปเสื้อ (สำหรับเช็คความแท้)",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_uploader",
        label_visibility="collapsed",
    )

    # Show preview if image uploaded
    if uploaded_file:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(uploaded_file, caption="รูปที่เลือก", use_container_width=True)
        with col2:
            st.caption("รูปพร้อมแล้ว — พิมพ์คำถามเพื่อให้ KitScout วิเคราะห์ครับ")

    # Pre-fill from quick example buttons
    prefill = st.session_state.get("prefill", "")
    if prefill:
        st.session_state["prefill"] = ""

    # Chat input
    user_input = st.chat_input("พิมพ์คำถาม เช่น 'เช็คความแท้จากรูปนี้ Arsenal 2002'") or prefill

    if user_input:
        # Prepare image for Gemini if uploaded
        pil_image = None
        image_bytes = None
        if uploaded_file:
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))

        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "image": image_bytes,
        })

        with st.chat_message("user"):
            if image_bytes:
                st.image(image_bytes, caption="รูปที่แนบมา", use_container_width=True)
            st.markdown(user_input)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("KitScout กำลังวิเคราะห์..."):
                answer = run_agent(
                    st.session_state.model,
                    st.session_state.chat,
                    user_input,
                    pil_image,
                )
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })

        st.rerun()


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
def main():
    init_session()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
