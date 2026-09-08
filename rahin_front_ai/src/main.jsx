import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Home as HomeIcon, LayoutDashboard, MessageSquareText, Tags, UsersRound,
  Boxes, Info, Phone, Settings, LogOut, ChevronLeft, ChevronDown, Download,
  Sun, Moon, Sparkles, ArrowUpLeft, ShieldCheck, TrendingUp, ShoppingCart,
  Eye, Star, Mail, RefreshCw, Menu, X, BrainCircuit, BookOpen,
  Send, Paperclip, BarChart3, Target, CircleHelp, Bot, Gem, UserRoundSearch, Pin, Plus, MoreHorizontal, MessageCircle, Clock3, Trash2 as TrashIcon,
  Layers3, Gauge, Headphones
} from "lucide-react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from "recharts";
import "./styles.css";

export const brand = {
  name: "راهین",
  subtitle: "سامانه هوش مشتریان",
  logo: "/assets/logo-mark.png"
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) }
  });
  if (!response.ok) throw new Error("خطا در دریافت اطلاعات");
  return response;
}
async function json(path, options = {}) {
  return (await api(path, options)).json();
}

const menu = [
  ["home", "خانه", HomeIcon],
  ["dashboard", "داشبورد", LayoutDashboard],
  ["assistant", "دستیار هوشمند", MessageSquareText],
  ["brand", "هوش برند", Tags],
  ["customers", "هوش مشتریان", UsersRound],
  ["categories", "تحلیل دسته‌بندی‌ها", Boxes],
  ["guide", "راهنما", BookOpen],
  ["about", "درباره ما", Info],
  ["contact", "ارتباط با ما", Phone],
  ["settings", "تنظیمات", Settings]
];

function App() {
  const [authenticated, setAuthenticated] = useState(localStorage.getItem("hooshyar-auth") === "1");
  const [page, setPage] = useState("home");
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "light");
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  const go = (next) => {
    setPage(next);
    setMobileOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const login = () => {
    localStorage.setItem("hooshyar-auth", "1");
    setAuthenticated(true);
    setPage("home");
  };

  const logout = () => {
    localStorage.removeItem("hooshyar-auth");
    setAuthenticated(false);
    setPage("home");
  };

  if (!authenticated) return <Login onLogin={login} />;

  return (
    <div className="app-shell">
      <Sidebar page={page} go={go} open={mobileOpen} close={() => setMobileOpen(false)} onLogout={logout} />
      <main className="main">
        <button className="mobile-menu" onClick={() => setMobileOpen(true)} aria-label="باز کردن منو"><Menu /></button>
        <div className="content">
          {page === "home" && <Home go={go} />}
          {page === "dashboard" && <Dashboard />}
          {page === "assistant" && <Assistant />}
          {page === "brand" && <BrandIntelligence />}
          {page === "customers" && <CustomerIntelligence />}
          {page === "categories" && <CategoryIntelligence />}
          {page === "guide" && <Guide go={go} />}
          {page === "about" && <About />}
          {page === "contact" && <Contact />}
          {page === "settings" && <SettingsPage theme={theme} setTheme={setTheme} />}
        </div>
      </main>
    </div>
  );
}

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  function submit(e) {
    e.preventDefault();
    if (username.trim() && password.trim()) onLogin();
  }

  return (
    <main className="login-page">
      <div className="login-glow login-glow-a" />
      <div className="login-glow login-glow-b" />
      <section className="login-card">
        <div className="login-art">
          <div className="login-art-orb"><img src={brand.logo} alt="" /></div>
          <div className="login-art-copy">
            <strong>{brand.name}</strong>
            <span>{brand.subtitle}</span>
            <p>داده‌ها را به بینش و بینش را به تصمیم تبدیل کنید.</p>
          </div>
          <div className="login-art-shape shape-one" />
          <div className="login-art-shape shape-two" />
        </div>
        <form className="login-form" onSubmit={submit}>
          <div className="login-logo"><img src={brand.logo} alt="" /></div>
          <span className="eyebrow">ورود به سامانه</span>
          <h1>خوش آمدید</h1>
          <p>برای ورود، اطلاعات حساب خود را وارد کنید.</p>
          <label>نام کاربری<input value={username} onChange={e => setUsername(e.target.value)} autoComplete="username" /></label>
          <label>رمز عبور<input type="password" value={password} onChange={e => setPassword(e.target.value)} autoComplete="current-password" /></label>
          <button className="login-submit" type="submit">ورود به سامانه <ChevronLeft /></button>
        </form>
      </section>
    </main>
  );
}

function Sidebar({ page, go, open, close, onLogout }) {
  return (
    <>
      {open && <button className="sidebar-overlay" onClick={close} aria-label="بستن منو" />}
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="side-brand">
          <img src={brand.logo} alt="" />
          <div>
            <strong>{brand.name}</strong>
            <span>{brand.subtitle}</span>
          </div>
          <button className="mobile-close" onClick={close}><X /></button>
        </div>

        <nav className="side-nav">
          {menu.map(([id, label, Icon]) => (
            <button
              key={id}
              className={`nav-item ${page === id ? "active" : ""} ${id === "dashboard" || id === "assistant" ? "key-nav" : ""}`}
              onClick={() => go(id)}
            >
              <Icon />
              <span>{label}</span>
              {(id === "dashboard" || id === "assistant") && <i />}
            </button>
          ))}
          <button className="nav-item logout-nav" onClick={onLogout}>
            <LogOut /><span>خروج</span>
          </button>
        </nav>

        <div className="side-promo">
          <div className="promo-orb" />
          <div className="promo-wave wave-a" />
          <div className="promo-wave wave-b" />
          <Sparkles />
          <strong>با راهین</strong>
          <span>یک قدم جلوتر...</span>
        </div>

      </aside>
    </>
  );
}

function Home({ go }) {
  return (
    <section className="home-page">
      <div className="home-hero">
        <div className="hero-copy">
          <div className="hero-logo"><img src={brand.logo} alt="" /></div>
          <h1>{brand.name}</h1>
          <h2>سامانه هوش مشتریان</h2>
          <p>با تحلیل داده‌های رفتاری، مسیر رشد کسب‌وکار خود را هوشمندانه‌تر بسازید.</p>
          <div className="hero-points">
            <span><BarChart3 /> تحلیل داده‌های واقعی</span>
            <span><Sparkles /> پیش‌بینی‌های هوشمند</span>
            <span><Target /> تصمیم‌گیری بهتر</span>
          </div>
        </div>
        <div className="hero-art"><img src="/assets/hero-art.png" alt="" /></div>
        <div className="hero-note">
          داده‌ها،<br />
          تصمیم‌های بهتر،<br />
          آینده‌ای روشن‌تر
          <i />
        </div>
      </div>

      <div className="home-primary">
        <HomeAction
          type="assistant"
          title="دستیار هوشمند"
          text="پرسش‌های خود را بپرسید، تحلیل‌های عمیق بگیرید و برای تصمیم‌های بهتر پاسخ دریافت کنید."
          action="شروع گفتگو"
          image="/assets/assistant-art.png"
          onClick={() => go("assistant")}
        />
        <HomeAction
          type="dashboard"
          title="داشبورد"
          text="نمای کلی عملکرد کسب‌وکار را با شاخص‌های کلیدی و نمودارهای تعاملی مشاهده کنید."
          action="مشاهده داشبورد"
          image="/assets/dashboard-art.png"
          onClick={() => go("dashboard")}
        />
      </div>

      <div className="home-quick-grid">
        <QuickCard icon={<Gem />} title="هوش برند" text="تحلیل جایگاه برند در ذهن مشتریان و بازار." onClick={() => go("brand")} tone="violet" />
        <QuickCard icon={<UsersRound />} title="هوش مشتریان" text="بخش‌بندی مشتریان و شناخت گروه‌های هدف." onClick={() => go("customers")} tone="blue" />
        <QuickCard icon={<Layers3 />} title="تحلیل دسته‌بندی‌ها" text="بررسی عملکرد دسته‌های مختلف از نظر فروش و رضایت." onClick={() => go("categories")} tone="pink" />
        <QuickCard icon={<Settings />} title="تنظیمات" text="ظاهر سامانه و حالت نمایش را مدیریت کنید." onClick={() => go("settings")} tone="blue" />
        <QuickCard icon={<Info />} title="درباره ما" text="با هدف و تیم پلتفرم هوشمند آشنا شوید." onClick={() => go("about")} tone="orange" />
        <QuickCard icon={<Phone />} title="ارتباط با ما" text="راه‌های ارتباط با تیم سامانه را مشاهده کنید." onClick={() => go("contact")} tone="green" />
      </div>

      <div className="home-bottom">
        <span><i /> سامانه آنلاین است</span>
        <div><span>{brand.name}</span><b>|</b><span>راهکاری برای مشتریان هوشمند</span></div>
      </div>
    </section>
  );
}

function HomeAction({ title, text, action, image, type, onClick }) {
  return (
    <button className={`home-action ${type}`} onClick={onClick}>
      <div className="action-art"><img src={image} alt="" /></div>
      <div className="action-copy">
        <h3>{title}</h3>
        <p>{text}</p>
        <span className="action-button">{action}<ChevronLeft /></span>
      </div>
    </button>
  );
}

function QuickCard({ icon, title, text, onClick, tone }) {
  return (
    <button className={`quick-card ${tone}`} onClick={onClick}>
      <div className="quick-icon">{icon}</div>
      <div><h3>{title}</h3><p>{text}</p></div>
      <span className="quick-arrow"><ArrowUpLeft /></span>
    </button>
  );
}

function PageTitle({ eyebrow, title, desc, icon: Icon = Sparkles }) {
  return (
    <div className="page-title">
      <div className="title-icon"><Icon /></div>
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{desc}</p>
      </div>
    </div>
  );
}

function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    json("/api/dashboard").then(setData).catch(() => setData(null)).finally(() => setLoading(false));
  }, []);
  const trend = data?.trend || [];
  const brands = data?.brands || [];
  const products = data?.products || [];
  const segments = data?.segments || [];

  return (
    <section className="dashboard-page">
      <PageTitle eyebrow="مرکز تصمیم‌گیری" title="داشبورد مدیریتی" desc="تصویری یکپارچه از رفتار مشتریان، فروش و عملکرد کسب‌وکار." icon={LayoutDashboard} />
      <div className="chart-card wide">
        <div className="card-head"><div><h3>روند کیفیت خرید ۳۰ روز اخیر</h3><span>مقایسه بازدید، سبد، خرید و حذف از سبد</span></div><RefreshCw /></div>
        {loading ? <ChartSkeleton /> : trend.length ? (
          <ResponsiveContainer width="100%" height={330}>
            <LineChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="views" name="بازدید" strokeWidth={3} />
              <Line type="monotone" dataKey="carts" name="سبد" strokeWidth={3} />
              <Line type="monotone" dataKey="purchases" name="خرید" strokeWidth={3} />
              <Line type="monotone" dataKey="removes" name="حذف از سبد" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        ) : <EmptyChart text="داده روند ۳۰ روز اخیر از سرویس داشبورد دریافت نشد." />}
      </div>
      <div className="three-charts">
        <ChartBlock title="پرفروش‌ترین برندها — ۳۰ روز اخیر" data={brands} keyName="brand_name" valueKey="total_purchases_30d" empty="اطلاعات برندها هنوز دریافت نشده است." />
        <ChartBlock title="گروه‌های رفتاری مشتریان" data={segments} keyName="segment" valueKey="count" empty="اطلاعات بخش‌بندی مشتریان هنوز دریافت نشده است." />
        <ChartBlock title="پرفروش‌ترین محصولات — ۳۰ روز اخیر" data={products} keyName="title_fa" valueKey="total_purchases_30d" empty="اطلاعات محصولات هنوز دریافت نشده است." />
      </div>
    </section>
  );
}

function Metric({ title, value, icon: Icon }) {
  return <div className="metric"><div className="metric-icon"><Icon /></div><div><span>{title}</span><strong>{value == null ? "—" : Number(value).toLocaleString("fa-IR")}</strong></div></div>;
}
function ChartBlock({ title, data, keyName, valueKey, empty }) {
  return <div className="chart-card small"><div className="card-head"><h3>{title}</h3></div>{data?.length ? (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart layout="vertical" data={data} margin={{ right: 15, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" />
        <YAxis type="category" dataKey={keyName} width={105} />
        <Tooltip />
        <Bar dataKey={valueKey} name="تعداد" radius={[0, 8, 8, 0]} />
      </BarChart>
    </ResponsiveContainer>
  ) : <EmptyChart text={empty} />}</div>;
}
function EmptyChart({ text }) { return <div className="empty-chart"><Sparkles /><span>{text}</span></div>; }
function ChartSkeleton() { return <div className="chart-skeleton"><span /><span /><span /></div>; }

const preparedQuestions = [
  "در ۶ ماه اخیر، ۱۰ محصول با بیشترین تعداد خرید کدام‌اند و هرکدام چند بار خریداری شده‌اند؟",
  "۱۰ محصول با بیشترین بازدید اما پایین‌ترین نرخ تبدیل به خرید کدام‌اند؟",
  "۱۰ برند برتر از نظر تعداد فروش و درآمد در ۳۰ روز اخیر کدام‌اند؟",
  "۱۰ دسته‌بندی برتر از نظر تعداد خرید و تعداد خریداران کدام‌اند؟",
  "۱۰ محصول با بیشترین بازخورد منفی مشتریان کدام‌اند و مهم‌ترین ویژگی‌های منفی آن‌ها چیست؟"
];

function Assistant() {
  const STORAGE_KEY = "rahin-chat-sessions";
  const [sessions, setSessions] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch { return []; }
  });
  const [activeId, setActiveId] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const persist = (next) => {
    setSessions(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  };

  const makeSession = () => ({
    id: `chat-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: "گفت‌وگوی جدید",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    pinned: false,
    messages: []
  });

  const createChat = () => {
    const s = makeSession();
    persist([s, ...sessions]);
    setActiveId(s.id);
    setInput("");
  };

  useEffect(() => {
    if (!sessions.length) {
      const s = makeSession();
      setSessions([s]);
      setActiveId(s.id);
      localStorage.setItem(STORAGE_KEY, JSON.stringify([s]));
    } else if (!activeId || !sessions.some(s => s.id === activeId)) {
      setActiveId(sessions[0].id);
    }
  }, []);

  const active = sessions.find(s => s.id === activeId) || sessions[0];
  const messages = active?.messages || [];
  const pinned = sessions.filter(s => s.pinned).sort((a,b) => b.updatedAt - a.updatedAt);
  const recent = sessions.filter(s => !s.pinned).sort((a,b) => b.updatedAt - a.updatedAt);

  const updateSession = (id, patch) => {
    const next = sessions.map(s => s.id === id ? { ...s, ...patch, updatedAt: Date.now() } : s);
    persist(next);
  };

  const togglePin = (id) => {
    const s = sessions.find(x => x.id === id);
    if (s) updateSession(id, { pinned: !s.pinned });
  };

  const deleteChat = (id) => {
    const next = sessions.filter(s => s.id !== id);
    if (!next.length) {
      const fresh = makeSession();
      persist([fresh]);
      setActiveId(fresh.id);
    } else {
      persist(next);
      if (id === activeId) setActiveId(next[0].id);
    }
  };

  async function ask(question) {
    const q = question.trim();
    if (!q || !active) return;
    setInput("");
    const now = Date.now();
    const userMessage = { role: "user", text: q, at: now };
    const current = sessions.find(s => s.id === active.id);
    const title = current?.messages?.length ? current.title : (q.length > 38 ? `${q.slice(0, 38)}…` : q);
    const withUser = sessions.map(s => s.id === active.id ? { ...s, title, messages: [...s.messages, userMessage], updatedAt: now } : s);
    persist(withUser);
    setLoading(true);
    try {
      const data = await json("/api/chat", { method: "POST", body: JSON.stringify({ message: q, session_id: active.id }) });
      const answer = data.answer || data.response || "پاسخی از سرویس دریافت نشد.";
      const latest = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      const next = latest.map(s => s.id === active.id ? { ...s, messages: [...s.messages, { role: "assistant", text: answer, at: Date.now() }], updatedAt: Date.now() } : s);
      persist(next);
    } catch {
      const latest = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      const next = latest.map(s => s.id === active.id ? { ...s, messages: [...s.messages, { role: "assistant", text: "اتصال به دستیار برقرار نشد. تنظیمات اتصال سرویس را بررسی کنید.", at: Date.now() }], updatedAt: Date.now() } : s);
      persist(next);
    } finally { setLoading(false); }
  }

  const selectQuestion = (q) => ask(q);

  return (
    <section className="assistant-page">
      <PageTitle eyebrow="تحلیل هوشمند" title="دستیار هوشمند" desc="سؤال مدیریتی خود را به زبان طبیعی بپرسید و پاسخ تحلیلی دریافت کنید." icon={MessageSquareText} />
      <div className="chat-workspace">
        <aside className="chat-sidebar">
          <button className="new-chat-btn" onClick={createChat}><Plus /> گفت‌وگوی جدید</button>
          <div className="chat-sidebar-section">
            <div className="chat-sidebar-label"><Pin /> پین‌شده‌ها</div>
            {pinned.length ? pinned.map(s => <ChatSessionItem key={s.id} session={s} active={s.id === activeId} onSelect={() => setActiveId(s.id)} onPin={() => togglePin(s.id)} onDelete={() => deleteChat(s.id)} />) : <div className="chat-sidebar-empty">گفت‌وگوی پین‌شده‌ای ندارید.</div>}
          </div>
          <div className="chat-sidebar-section recent-section">
            <div className="chat-sidebar-label"><Clock3 /> گفت‌وگوهای اخیر</div>
            {recent.map(s => <ChatSessionItem key={s.id} session={s} active={s.id === activeId} onSelect={() => setActiveId(s.id)} onPin={() => togglePin(s.id)} onDelete={() => deleteChat(s.id)} />)}
          </div>
        </aside>

        <div className="chat-main">
          <div className="chat-topbar">
            <div className="chat-topbar-badge"><Bot /></div>
            <div className="chat-topbar-info"><strong>{active?.title || "گفت‌وگوی جدید"}</strong></div>
            <div className="chat-status"><i /> آنلاین</div>
          </div>

          {!messages.length && !input.trim() && (
            <div className="chat-welcome">
              <div className="chat-welcome-bot"><img src="/assets/assistant-art.png" alt="دستیار راهین" /></div>
              <h2>از داده‌ها سؤال بپرسید</h2>
              <p>یکی از پرسش‌های آماده را انتخاب کنید یا سؤال مدیریتی خودتان را بنویسید.</p>
              <div className="question-grid question-grid-large">
                {preparedQuestions.map((q, i) => <button key={q} onClick={() => selectQuestion(q)}><span className="question-num">{String(i + 1).padStart(2, "۰")}</span><span>{q}</span><ChevronLeft /></button>)}
              </div>
            </div>
          )}

          {messages.length > 0 && (
            <div className="compact-prompts">
              {preparedQuestions.map(q => <button key={q} onClick={() => selectQuestion(q)}>{q}</button>)}
            </div>
          )}

          <div className="chat-messages chat-messages-large">
            {messages.map((m, i) => <div key={i} className={`message ${m.role}`}>{m.text}</div>)}
            {loading && <div className="message assistant"><span className="typing-dots">در حال تحلیل اطلاعات<span>.</span><span>.</span><span>.</span></span></div>}
          </div>
          <div className="composer composer-large">
            <button title="افزودن فایل"><Paperclip /></button>
            <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === "Enter" && ask(input)} placeholder="سؤال مدیریتی خود را بنویسید..." />
            <button className="send" onClick={() => ask(input)} disabled={loading}><Send /></button>
          </div>
        </div>
      </div>
    </section>
  );
}

function ChatSessionItem({ session, active, onSelect, onPin, onDelete }) {
  return <div className={`chat-session ${active ? "active" : ""}`} onClick={onSelect}>
    <div className="chat-session-icon"><MessageCircle /></div>
    <div className="chat-session-text"><strong>{session.title}</strong></div>
    <div className="chat-session-actions">
      <button title={session.pinned ? "برداشتن پین" : "پین کردن"} onClick={e => { e.stopPropagation(); onPin(); }}><Pin /></button>
      <button title="حذف" onClick={e => { e.stopPropagation(); onDelete(); }}><TrashIcon /></button>
    </div>
  </div>;
}

function CustomerIntelligence() {
  const [rows, setRows] = useState([]);
  useEffect(() => { json("/api/customer-segments").then(d => setRows(Array.isArray(d) ? d : d.items || [])).catch(() => {}); }, []);
  const fallback = [
    { segment: "VIP Customer" }, { segment: "Returning Customer" }, { segment: "One-Time Buyer" },
    { segment: "Low Engagement" }, { segment: "Window Shopper" }
  ];
  return (
    <section>
      <PageTitle eyebrow="تحلیل مشتری" title="هوش مشتریان" desc="بخش‌های مختلف مشتریان را بشناسید و فهرست هر بخش را برای تحلیل بیشتر دریافت کنید." icon={UsersRound} />
      <div className="info-banner"><ShieldCheck /><div><strong>بخش‌بندی متمرکز بر اقدام</strong><span>برای هر بخش امکان دریافت فهرست مشتریان به صورت اکسل وجود دارد.</span></div></div>
      <div className="table-card"><div className="card-head"><div><h3>بخش‌های مشتریان</h3><span>تعداد، سهم و دریافت فهرست هر بخش</span></div></div>
        <div className="table-wrap"><table><thead><tr><th>بخش مشتری</th><th>تعداد مشتری</th><th>سهم از مشتریان</th><th>عملیات</th></tr></thead>
          <tbody>{(rows.length ? rows : fallback).map((r, i) => <tr key={i}><td><span className={`segment-dot segment-${i % 5}`} />{translateSegment(r.segment)}</td><td>{r.count == null ? "—" : fmt(r.count)}</td><td>{r.share == null ? "—" : `${fmt(r.share)}٪`}</td><td><button className={`download download-${i % 5}`} onClick={() => downloadSegment(r.segment)}><Download /> دریافت اکسل</button></td></tr>)}</tbody>
        </table></div>
      </div>
    </section>
  );
}
function translateSegment(s) {
  return ({ "VIP Customer": "مشتری ویژه", "Returning Customer": "مشتری بازگشتی", "One-Time Buyer": "خریدار تک‌مرتبه‌ای", "Low Engagement": "تعامل پایین", "Window Shopper": "بازدیدکننده بدون خرید", "Window Shopper (فقط بازدیدکننده)": "بازدیدکننده بدون خرید" })[s] || s || "—";
}
async function downloadSegment(s) {
  try {
    const map = { "VIP Customer": "vip", "Returning Customer": "returning", "One-Time Buyer": "one-time", "Low Engagement": "low-engagement", "Window Shopper": "window-shopper" };
    const response = await api(`/api/customer-segments/${map[s] || encodeURIComponent(s)}/export`);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob); const a = document.createElement("a");
    a.href = url; a.download = `${map[s] || "customers"}.xlsx`; a.click(); URL.revokeObjectURL(url);
  } catch { alert("دریافت فایل از سرویس امکان‌پذیر نیست."); }
}

function BrandIntelligence() {
  const [data, setData] = useState([]);
  useEffect(() => { json("/api/brand-intelligence").then(d => setData(d.items || d || [])).catch(() => {}); }, []);
  return <section><PageTitle eyebrow="تحلیل برند" title="هوش برند" desc="ارزیابی جایگاه برندها بر اساس تعامل، فروش، درآمد و بازخورد مشتریان." icon={Tags} />
    <div className="insight-grid"><Insight icon={Eye} title="تعامل" text="بازدید و تعامل برندها را مقایسه کنید." tone="blue" /><Insight icon={TrendingUp} title="فروش" text="عملکرد فروش برندها را کنار هم ببینید." tone="violet" /><Insight icon={Star} title="رضایت" text="بازخورد و احساسات مشتریان را بررسی کنید." tone="orange" /></div>
    <div className="table-card"><div className="card-head"><div><h3>عملکرد برندها</h3><span>اطلاعات از سرویس هوش برند دریافت می‌شود.</span></div></div>
      <div className="table-wrap"><table><thead><tr><th>برند</th><th>بازدید</th><th>فروش</th><th>درآمد</th><th>نرخ تبدیل</th><th>رضایت</th></tr></thead>
        <tbody>{data.length ? data.map((r, i) => <tr key={i}><td>{r.brand_name || "—"}</td><td>{fmt(r.total_views_30d ?? r.total_views)}</td><td>{fmt(r.total_purchases_30d ?? r.total_purchases)}</td><td>{money(r.total_revenue_30d)}</td><td>{pct(r.conversion_rate_30d)}</td><td>{r.brand_sentiment_score == null ? "—" : r.brand_sentiment_score}</td></tr>) : <tr><td colSpan="6" className="no-data">داده‌ای برای نمایش دریافت نشد.</td></tr>}</tbody>
      </table></div></div>
  </section>;
}
function Insight({ icon: Icon, title, text, tone }) { return <div className={`insight ${tone}`}><div><Icon /></div><h3>{title}</h3><p>{text}</p></div>; }

function CategoryIntelligence() {
  const [cats, setCats] = useState([]);
  useEffect(() => { json("/api/category-intelligence").then(d => setCats(d.items || d || [])).catch(() => {}); }, []);
  const defaults = ["beauty", "book & stationary & art", "clothe", "rural goods", "toys and kids", "travel"];
  return <section><PageTitle eyebrow="تحلیل دسته‌بندی" title="تحلیل دسته‌بندی‌ها" desc="شش دسته اصلی را از منظر فروش، تبدیل و رضایت بررسی کنید." icon={Boxes} />
    <div className="category-grid">{(cats.length ? cats : defaults.map(name => ({ name }))).map((c, i) => <CategoryCard key={i} c={c} index={i} />)}</div>
  </section>;
}
function CategoryCard({ c, index }) {
  const icons = [Gem, BookOpen, Tags, Layers3, Boxes, Target];
  const Icon = icons[index] || Boxes;
  return <div className="category-card"><div className="cat-top"><span className="catnum">{String(index + 1).padStart(2, "۰")}</span><div className="cat-icon"><Icon /></div></div><h3>{translateCat(c.name)}</h3>
    <div className="status-list"><Status label="فروش" value={statusText(c.sales_status || c.sales)} /><Status label="تبدیل" value={statusText(c.conversion_status || c.conversion)} /><Status label="رضایت" value={statusText(c.satisfaction_status || c.satisfaction)} /></div>
  </div>;
}
function translateCat(x) { return ({ beauty: "زیبایی و سلامت", "book & stationary & art": "کتاب، نوشت‌افزار و هنر", clothe: "پوشاک", "rural goods": "کالای روستایی", "toys and kids": "اسباب‌بازی و کودک", travel: "سفر" })[x] || x; }
function statusText(v) { if (!v) return "—"; return ({ strong: "قوی", medium: "متوسط", weak: "ضعیف", good: "خوب", low: "پایین" })[String(v).toLowerCase()] || v; }
function Status({ label, value }) { return <div className="status"><span>{label}</span><b>{value}</b></div>; }

function Guide({ go }) {
  const sections = [
    ["شروع کار با سامانه", "از صفحه خانه مسیر مورد نیاز خود را انتخاب کنید. داشبورد برای مشاهده وضعیت کلی، صفحات تحلیلی برای بررسی جزئیات و دستیار هوشمند برای پاسخ به پرسش‌های مدیریتی طراحی شده‌اند.", "/assets/guide-home.svg"],
    ["داشبورد مدیریتی", "در داشبورد روند ۳۰ روز اخیر، عملکرد برندها، مشتریان و محصولات را در یک نمای یکپارچه مشاهده کنید. نمودارها از سرویس داده سامانه دریافت می‌شوند.", "/assets/dashboard-art.png"],
    ["دستیار هوشمند", "سؤال خود را به زبان طبیعی مطرح کنید یا یکی از پنج سؤال آماده را انتخاب کنید. پاسخ‌ها می‌توانند برای بررسی محصولات، برندها، دسته‌بندی‌ها و بازخورد مشتریان استفاده شوند.", "/assets/assistant-art.png"],
    ["هوش مشتریان", "مشتریان در بخش‌های مختلف دسته‌بندی می‌شوند و برای هر بخش امکان دریافت فهرست اکسل وجود دارد. این بخش برای شناخت گروه‌های ارزشمند، بازگشتی و کم‌تعامل کاربرد دارد.", "/assets/guide-customers.svg"],
    ["هوش برند", "در این بخش عملکرد برندها از منظر تعامل، فروش، درآمد و بازخورد مشتریان مقایسه می‌شود تا نقاط قوت و فرصت‌های بهبود سریع‌تر مشخص شوند.", "/assets/guide-brand.svg"],
    ["تحلیل دسته‌بندی‌ها", "شش دسته اصلی سامانه از نظر فروش، نرخ تبدیل و رضایت بررسی می‌شوند. این صفحه به مدیر کمک می‌کند عملکرد دسته‌ها را سریع و قابل مقایسه ببیند.", "/assets/guide-category.svg"]
  ];
  return <section className="guide-page">
    <PageTitle eyebrow="راهنمای استفاده" title="راهنما" desc="با بخش‌های مختلف سامانه آشنا شوید و مسیر مناسب تحلیل را پیدا کنید." icon={BookOpen} />
    <div className="guide-hero">
      <div><span className="soft-label"><BookOpen /> راهنمای راهین</span><h2>همه‌چیز برای شروع، همین‌جاست</h2><p>بخش مورد نظر را باز کنید تا کاربردها و روش استفاده از آن را به زبان ساده ببینید.</p></div>
      <img src="/assets/guide-hero.svg" alt="" />
    </div>
    <div className="guide-grid">{sections.map(([title, text, img], i) => <details key={title} open={i === 0}><summary><span className="guide-number">{String(i + 1).padStart(2, "۰")}</span><span>{title}</span><ChevronDown /></summary><div className="guide-body"><div className="guide-art"><img src={img} alt="" /></div><div><p>{text}</p>{i === 2 && <button className="guide-action" onClick={() => go("assistant")}>رفتن به دستیار هوشمند <ChevronLeft /></button>}</div></div></details>)}</div>
  </section>;
}

function About() {

  const teamMembers = ["ریحانه شریفی","زینب عابدینی","فاطمه کلهری"];

  return (
    <section>
      <PageTitle 
        eyebrow="درباره سامانه" 
        title="درباره ما" 
        desc="پلتفرمی برای تبدیل داده‌های رفتاری مشتریان به بینش قابل اقدام."
        icon={Info}
      />

      <div className="about-hero">
        <div className="about-art">
          <img src="/assets/about-art.svg" alt="" />
        </div>

        <div>
          <span className="soft-label">
            <Sparkles /> راهین
          </span>

          <h2>پلتفرم هوشمند تحلیل رفتار مشتریان</h2>

          <p>
            این پلتفرم با ترکیب داده‌های رفتاری، شاخص‌های کلیدی عملکرد و قابلیت‌های هوش مصنوعی، اطلاعات خام مشتریان را به اطلاعات قابل فهم و قابل استفاده برای تصمیم‌گیری تبدیل می‌کند.
          </p>

          <p>
            هدف سامانه، کمک به مدیران برای شناخت بهتر رفتار مشتری، ارزیابی عملکرد محصول و برند، پاسخ‌گویی به پرسش‌های کسب‌وکار و حرکت به سمت تصمیم‌گیری داده‌محور است.
          </p>

          <div className="goal">
            تبدیل داده به بینش، و بینش به تصمیم.
          </div>
        </div>
      </div>


      <h3 className="section-heading">تیم ما</h3>

      <div className="team-grid">
        {teamMembers.map((member, index) => (
          <div className="team-card" key={index}>
            <div className="person">
              <UserRoundSearch />
            </div>

            <div>
              <h3>{member}</h3>
            </div>
          </div>
        ))}
      </div>

    </section>
  );
}

function Contact() {
  return <section><PageTitle eyebrow="ارتباط" title="ارتباط با ما" desc="برای ارتباط با تیم سامانه از راه‌های زیر استفاده کنید." icon={Phone} />
    <div className="contact-simple"><div className="contact-item"><div><Mail /></div><span>ایمیل</span><a href="mailto:info@example.com">info@example.com</a></div><div className="contact-item"><div><Phone /></div><span>شماره تماس</span><a href="tel:+980000000000">۰۹۱۲ ۰۰۰ ۰۰۰۰</a></div></div>
  </section>;
}

function SettingsPage({ theme, setTheme }) {
  return <section><PageTitle eyebrow="شخصی‌سازی" title="تنظیمات" desc="حالت نمایش سامانه را مطابق سلیقه خود تنظیم کنید." icon={Settings} />
    <div className="settings-card"><div className="settings-art"><Gauge /></div><div><h3>حالت نمایش</h3><p>بین حالت روشن و شب انتخاب کنید.</p><div className="theme-switch"><button className={theme === "light" ? "active" : ""} onClick={() => setTheme("light")}><Sun /> حالت روشن</button><button className={theme === "dark" ? "active" : ""} onClick={() => setTheme("dark")}><Moon /> حالت شب</button></div></div></div>
  </section>;
}

function fmt(v) { return v == null ? "—" : Number(v).toLocaleString("fa-IR"); }
function pct(v) { return v == null ? "—" : `${Number(v).toLocaleString("fa-IR")}٪`; }
function money(v) { return v == null ? "—" : `${Number(v).toLocaleString("fa-IR")} تومان`; }

createRoot(document.getElementById("root")).render(<App />);
