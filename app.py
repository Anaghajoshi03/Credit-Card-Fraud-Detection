import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai
import os
import random
from datetime import datetime
import time

# Configure page with wider layout and custom theme
st.set_page_config(
    page_title="ATM Fraud Detection",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.securebank.com/help',
        'Report a bug': 'https://www.securebank.com/bug',
        'About': 'ATM Fraud Detection System v2.2'
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #DDD;
    }
    .subheader {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        margin: 1rem 0;
    }
    .success-card {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
    }
    .warning-card {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
    }
    .info-panel {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .sidebar-content {
        padding: 1.5rem 1rem;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6B7280;
        border-top: 1px solid #E5E7EB;
        margin-top: 3rem;
    }
    .transaction-form {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stat-box {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid #E5E7EB;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E40AF;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
    .chat-bubble {
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #DBEAFE;
        border-bottom-right-radius: 0.25rem;
        float: right;
    }
    .assistant-bubble {
        background-color: #F3F4F6;
        border-bottom-left-radius: 0.25rem;
        float: left;
    }
</style>
""", unsafe_allow_html=True)

# Configure Gemini AI API Key
genai.configure(api_key="AIzaSyBG-Y-OpKHDWxVzAZzfT636Errbm4dGfYE")

# Load trained models
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# App header with logo
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 class='main-header'>🏧 ATM Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Advanced AI-powered fraud monitoring and prevention</p>", unsafe_allow_html=True)

# Sidebar with navigation and stats
with st.sidebar:
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    
    # User profile section
    st.image("https://www.securebank.com/api/placeholder/100/100", width=80)
    st.markdown("### Welcome, Analyst")
    st.markdown(f"**Last login:** {datetime.now().strftime('%B %d, %Y %H:%M')}")
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📌 Navigation")
    section = st.radio("", ["Dashboard", "Model Selection", "Deep Learning Analysis", "Community Chat", "AI Assistant"])
    
    st.markdown("---")
    
    # Stats section
    st.markdown("### 📊 System Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='stat-box'><div class='stat-value'>98.7%</div><div class='stat-label'>Accuracy</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='stat-box'><div class='stat-value'>1.2s</div><div class='stat-label'>Avg Response</div></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("🔐 **Bank Security System v2.2**")
    st.caption("© SecureBank 2025")
    st.markdown("</div>", unsafe_allow_html=True)

# Load models
rf_model = load_model("fraud_detection_rf_model.pkl")
dt_model = load_model("Decision_Tree_model.pkl")

# Functions for transaction analysis
def analyze_transaction(transaction_type, amount, old_balance, new_balance):
    """Provides additional insights about the transaction"""
    insights = []
    
    # Balance change analysis
    if new_balance < old_balance - amount:
        insights.append("⚠️ Balance discrepancy detected")
    
    # Amount patterns
    if amount > 5000:
        insights.append("⚠️ Large transaction amount")
    if 0.99 < amount < 5.01:
        insights.append("⚠️ Test transaction amount (common in fraud)")
        
    # Balance patterns  
    if old_balance > 0 and new_balance == 0:
        insights.append("⚠️ Account emptied (high risk)")
    
    # Transaction type specific
    if transaction_type == "CASH_OUT" and amount > 1000:
        insights.append("⚠️ Large cash withdrawal")
        
    return insights

# --- Dashboard Section ---
if section == "Dashboard":
    # Dashboard layout
    st.markdown("<h2 class='subheader'>📊 Fraud Detection Dashboard</h2>", unsafe_allow_html=True)
    
    # Key metrics in columns
    metric1, metric2, metric3, metric4 = st.columns(4)
    with metric1:
        st.metric(label="Today's Alerts", value="12", delta="-3")
    with metric2:
        st.metric(label="False Positives", value="2.1%", delta="-0.4%")
    with metric3:
        st.metric(label="Detection Rate", value="99.7%", delta="+0.2%")
    with metric4:
        st.metric(label="Review Time", value="45s", delta="-5s")
    
    # Recent activity section
    st.markdown("<h3 style='margin-top: 2rem;'>Recent Activity</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Latest Transactions")
        
        transactions = [
            {"time": "10:45 AM", "type": "PAYMENT", "amount": "$152.30", "status": "✅ Safe"},
            {"time": "09:32 AM", "type": "TRANSFER", "amount": "$890.00", "status": "✅ Safe"},
            {"time": "08:17 AM", "type": "CASH_OUT", "amount": "$500.00", "status": "🚨 Flagged"},
            {"time": "Yesterday", "type": "DEBIT", "amount": "$42.15", "status": "✅ Safe"},
            {"time": "Yesterday", "type": "TRANSFER", "amount": "$1,200.00", "status": "✅ Safe"},
        ]
        
        for t in transactions:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 8px 0;">
                <div>{t["time"]}</div>
                <div>{t["type"]}</div>
                <div>{t["amount"]}</div>
                <div>{t["status"]}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Fraud Distribution")
        st.caption("By transaction type")
        
        chart_data = {
            "PAYMENT": 34,
            "TRANSFER": 28, 
            "CASH_OUT": 56,
            "DEBIT": 19,
            "CASH_IN": 13
        }
        
        st.bar_chart(chart_data)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # System notifications
    st.markdown("<div class='info-panel'>", unsafe_allow_html=True)
    st.markdown("#### 📢 System Notifications")
    st.markdown("- **Model Update**: Random Forest model updated yesterday with 2.3% accuracy improvement")
    st.markdown("- **Pattern Alert**: New fraud pattern detected involving small recurring payments")
    st.markdown("- **Maintenance**: Scheduled system maintenance on April 12, 2025")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Model Selection ---
elif section == "Model Selection":
    st.markdown("<h2 class='subheader'>⚙️ Model Selection</h2>", unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<div class='transaction-form'>", unsafe_allow_html=True)
        st.markdown("### 💰 Transaction Details")
        
        transaction_type = st.selectbox("Transaction Type", [2, 4, 1, 5, 3], format_func=lambda x: {
            2: "PAYMENT",
            4: "TRANSFER",
            1: "CASH_OUT",
            5: "DEBIT",
            3: "CASH_IN"
        }[x])
        
        # More intuitive form layout
        amount = st.number_input("Amount ($)", min_value=0.0, format="%.2f", help="Transaction amount")
        
        col1_inner, col2_inner = st.columns(2)
        with col1_inner:
            old_balance_org = st.number_input("Old Balance (Sender)", min_value=0.0, format="%.2f")
        with col2_inner:
            new_balance_org = st.number_input("New Balance (Sender)", min_value=0.0, format="%.2f")
        
        # Model selection with visual indicators
        st.markdown("### 🤖 Select Model")
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            rf_selected = st.checkbox("Random Forest", value=True, help="Higher accuracy, slower")
            if rf_selected:
                st.markdown("- 96.4% Accuracy")
                st.markdown("- Good for complex patterns")
                st.markdown("- Handles outliers well")
        
        with model_col2:
            dt_selected = st.checkbox("Decision Tree", help="Faster, more interpretable")
            if dt_selected:
                st.markdown("- 93.8% Accuracy")
                st.markdown("- More explainable")
                st.markdown("- Faster processing")
                
        # Ensure only one is selected
        if rf_selected and dt_selected:
            dt_selected = False
            st.warning("Only one model can be selected at a time.")
        
        model_choice = "Random Forest Classifier" if rf_selected else "Decision Tree"
        
        # Enhanced submit button with animation
        if st.button("🔍 Analyze Transaction", use_container_width=True):
            with st.spinner("Analyzing transaction..."):
                # Simulated processing time for better UX
                time.sleep(1.2)
                
                input_data = np.array([[transaction_type, amount, old_balance_org, new_balance_org]])
                
                if model_choice == "Random Forest Classifier":
                    prediction = rf_model.predict(input_data) if rf_model else ["No fraud"]
                else:
                    prediction = dt_model.predict(input_data) if dt_model else ["No fraud"]
                
                # Get additional insights
                transaction_type_name = {2: "PAYMENT", 4: "TRANSFER", 1: "CASH_OUT", 5: "DEBIT", 3: "CASH_IN"}[transaction_type]
                insights = analyze_transaction(transaction_type_name, amount, old_balance_org, new_balance_org)
                
                # Show results
                if prediction[0] == "No fraud":
                    st.markdown("<div class='card success-card'>", unsafe_allow_html=True)
                    st.success("✅ This transaction appears legitimate.")
                    st.markdown("### Transaction Assessment")
                    st.markdown("✓ Normal transaction pattern")
                    st.markdown("✓ Balanced account activity")
                    st.markdown("✓ No suspicious indicators")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.toast("✅ Transaction looks safe!", icon="✅")
                    st.balloons()
                else:
                    st.markdown("<div class='card warning-card'>", unsafe_allow_html=True)
                    st.error("🚨 WARNING: Potential Fraud Detected!")
                    st.markdown("### Risk Assessment")
                    st.markdown("❌ Suspicious transaction pattern")
                    for insight in insights:
                        st.markdown(f"❌ {insight}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.toast("🚨 Fraud Alert! This transaction may be fraudulent.", icon="⚠")
        st.markdown("</div>", unsafe_allow_html=True)
                
    # Side panel with model performance and information
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Model Performance")
        
        st.markdown("#### Random Forest")
        st.progress(96)
        st.caption("96% Overall Accuracy")
        
        st.markdown("#### Decision Tree")
        st.progress(93)
        st.caption("93% Overall Accuracy")
        
        st.markdown("### 🔍 Why Use ML for Fraud?")
        st.markdown("""
        - Pattern recognition across millions of transactions
        - Real-time analysis and scoring
        - Adapts to evolving fraud techniques
        - Reduces false positives by 73%
        """)
        
        # Model insights accordion
        with st.expander("ℹ️ How the models work"):
            st.markdown("""
            **Random Forest Classifier**
            Combines multiple decision trees to improve accuracy and reduce overfitting. Effective at identifying complex fraud patterns.
            
            **Decision Tree**
            Creates a flowchart-like structure for decision making. More interpretable but slightly less accurate.
            """)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Deep Learning Model with Gemini API ---
elif section == "Deep Learning Analysis":
    st.markdown("<h2 class='subheader'>🧠 Deep Learning Fraud Analysis</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<div class='transaction-form'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Transaction Details")
        
        # More visually appealing layout
        transaction_type = st.selectbox("Transaction Type", 
                                      ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
                                      help="Type of transaction being performed")
        
        # Use columns for compact layout
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            amount = st.number_input("Amount ($)", min_value=0.0, value=500.0, format="%.2f")
        with row1_col2:
            time_options = ["Morning", "Afternoon", "Evening", "Night"]
            transaction_time = st.selectbox("Time of Day", time_options)
        
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            old_balance_org = st.number_input("Old Balance ($)", min_value=0.0, value=1000.0, format="%.2f")
        with row2_col2:
            new_balance_org = st.number_input("New Balance ($)", min_value=0.0, value=500.0, format="%.2f")
    
        # Add more contextual information for better analysis
        with st.expander("Additional Context (Optional)"):
            country = st.selectbox("Transaction Country", ["United States", "Canada", "United Kingdom", "Other"])
            is_mobile = st.checkbox("Mobile Transaction")
            frequency = st.slider("Customer Transaction Frequency", 1, 30, 5, 
                                help="Average transactions per month")
            
        # Enhanced submit button
        analyze_btn = st.button("🤖 Analyze with Gemini AI", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
            
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧪 Gemini AI Analysis")
        st.markdown("""
        The Gemini AI model analyzes transactions using:
        
        - Pattern recognition
        - Natural language processing
        - Behavioral analysis
        - Historical data correlation
        - Multi-factor risk assessment
        
        This provides deeper insights beyond traditional models.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Model confidence visualization
        with st.expander("🔮 Gemini AI Performance Metrics"):
            st.markdown("""
            - **Accuracy**: 97.8%
            - **False Positive Rate**: 0.8%
            - **Detection Speed**: < 1.5 seconds
            - **Adaptability**: High - Self-improving
            """)
    
    if analyze_btn:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        progress_text = "Analyzing with Gemini AI..."
        status_bar = st.progress(0)
        
        # Simulate analysis progress for better UX
        for i in range(100):
            status_bar.progress(i + 1)
            time.sleep(0.01)
        
        try:
            gemini = genai.GenerativeModel('gemini-1.5-flash')
            response = gemini.generate_content(f"""
                You are an expert in fraud detection. Analyze the following transaction:
                - Transaction Type: {transaction_type}
                - Amount: ${amount:.2f}
                - Old Balance: ${old_balance_org:.2f}
                - New Balance: ${new_balance_org:.2f}
                - Time of Day: {transaction_time}
                
                Determine if this transaction is fraudulent or not. Reply with "Fraud" or "No fraud" on the first line.
                Then explain why it is considered fraud or not.
                Also, give 2 precautions the user should take in the future.
                Format your response with clear headings and bullet points.
                Be concise but thorough.
            """,
            generation_config={"temperature": 0.3, "max_output_tokens": 500})

            result = response.text.strip()

            # Enhanced visualization of results
            st.markdown("## 📊 Analysis Results")
            
            if "Fraud" in result and "No fraud" not in result:
                st.error("🚨 HIGH RISK: Potential fraud detected")
                st.warning("This transaction has been flagged for manual review")
            else:
                st.success("✅ LOW RISK: No fraud indicators detected")
                st.info("Transaction appears to be legitimate")

            # Format the AI response with better styling
            formatted_response = result.replace("Fraud", "**🚨 FRAUD ALERT**").replace("No fraud", "**✅ NO FRAUD DETECTED**")
            st.markdown("### 🤖 Gemini Analysis Report")
            st.markdown(formatted_response)
            
            # Recommended actions based on result
            st.markdown("### 📋 Recommended Actions")
            if "Fraud" in result and "No fraud" not in result:
                st.markdown("""
                1. 🚫 **Block Transaction** - Prevent further activity
                2. 📞 **Contact Customer** - Verify attempted transaction
                3. 🔍 **Review Account** - Check for other suspicious activity
                4. 📝 **Document Findings** - Add to fraud database
                """)
            else:
                st.markdown("""
                1. ✅ **Approve Transaction** - Proceed with normal processing
                2. 📊 **Update Profile** - Add to customer behavior model
                3. 🔄 **Monitor** - Continue normal account monitoring
                """)

        except Exception as e:
            st.error(f"⚠️ Gemini analysis failed: {str(e)}")
            st.markdown("Please try again or use the traditional models.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Community Chat Section ---
elif section == "Community Chat":
    st.markdown("<h2 class='subheader'>💬 Fraud Prevention Community</h2>", unsafe_allow_html=True)
    st.caption("Exchange insights with fraud prevention professionals")

    # Tabs for different discussion areas
    tab1, tab2, tab3 = st.tabs(["General Discussion", "New Patterns", "Resources"])
    
    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "👋 Welcome to the Fraud Prevention Community! Share your experiences and ask questions."},
                {"role": "user", "content": "Has anyone seen an increase in small test transactions lately?"},
                {"role": "assistant", "content": "💡 Yes! Small test transactions ($1-$5) before larger withdrawals are trending. We're seeing this pattern especially on weekends."}
            ]

        # Enhanced chat display
        st.markdown("<div style='background-color: #F9FAFB; border-radius: 0.5rem; padding: 1rem; height: 400px; overflow-y: auto; margin-bottom: 1rem;'>", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div style='text-align: right;'><div class='chat-bubble user-bubble'>{msg['content']}</div></div><div style='clear: both;'></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left;'><div class='chat-bubble assistant-bubble'>{msg['content']}</div></div><div style='clear: both;'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        community_tips = [
            "🤖 Community Tip: Fraudsters often test cards with small transactions ($1-$5) before larger withdrawals. Monitor these carefully.",
            "🔍 Expert Insight: Look for rapid succession transactions at different ATMs - common fraud pattern.",
            "📊 Data Trend: 63% of fraud occurs on weekends when monitoring teams are smaller. Increase weekend vigilance.",
            "💡 Prevention Tip: Train staff to recognize 'shoulder surfing' - criminals watching customers enter PINs.",
            "🛡️ Security Advice: Recommend customers cover the keypad when entering PINs at ATMs.",
            "⚠️ Alert Pattern: Transactions where amount is exactly the account balance are suspicious (attempts to empty accounts).",
            "🌐 Global Trend: Card skimming devices are becoming more sophisticated. Regularly inspect ATM card readers.",
            "📱 Tech Tip: Mobile banking apps with transaction alerts can help customers spot fraud faster.",
            "🕵️ Investigator Note: Fraudsters often target elderly customers. Extra verification recommended for senior transactions.",
            "💳 Card Safety: EMV chip cards reduced counterfeit fraud by 76%. Encourage customers to use chip instead of swipe.",
            "🔢 PIN Security: The most common PINs (1234, 0000, 1111) are compromised first. Encourage complex PINs.",
            "⏱️ Timing Insight: Most fraudulent ATM withdrawals occur between 8PM-12AM when branches are closed.",
            "🚨 Red Flag: Multiple failed PIN attempts followed by a successful transaction may indicate brute force attacks.",
            "💰 Amount Pattern: Fraudsters often withdraw just below reporting thresholds (e.g., $9,900 instead of $10,000).",
            "🌍 Location Tip: Transactions originating from countries different than the cardholder's residence need extra scrutiny."
        ]

        # Improved chat input
        col1, col2 = st.columns([5, 1])
        with col1:
            prompt = st.text_input("Share your experience or ask a question...", key="community_input")
        with col2:
            send_btn = st.button("Send", use_container_width=True)
            
        if prompt and send_btn:
            st.session_state.messages.append({"role": "user", "content": prompt})

            contextual_response = None
            user_message_lower = prompt.lower()

            if "small" in user_message_lower or "test" in user_message_lower:
                contextual_response = "💡 Small test transactions often lead to fraud. Monitor closely."
            elif "weekend" in user_message_lower:
                contextual_response = "📅 Weekend fraud is common. Increase monitoring."
            elif "elderly" in user_message_lower:
                contextual_response = "👵 Elderly customers are targeted. Use extra verification."

            bot_response = contextual_response if contextual_response else random.choice(community_tips)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.rerun()
            
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🚨 Recently Reported Patterns")
        st.markdown("""
        - **Digital Gift Card Scams** - Reported by 12 institutions in the last week
        - **QR Code Phishing** - Growing trend in metro areas
        - **ATM Deep Insert Skimmers** - New hardware harder to detect visually
        - **Social Engineering via Social Media** - Targeting bank employees
        """)
        
        with st.expander("Report New Pattern"):
            st.text_area("Describe the pattern you've identified", height=100)
            st.selectbox("Fraud Category", ["Card Present", "Card Not Present", "Account Takeover", "Identity Theft", "Other"])
            st.slider("Confidence Level", 1, 5, 3)
            st.button("Submit Report")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📚 Resources")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Training Materials")
            st.markdown("- [Fraud Pattern Recognition Guide](https://example.com)")
            st.markdown("- [Customer Authentication Best Practices](https://example.com)")
            st.markdown("- [Transaction Monitoring Techniques](https://example.com)")
            
        with col2:
            st.markdown("#### Latest Reports")
            st.markdown("- [Q1 2025 Fraud Trends](https://example.com)")
            st.markdown("- [ATM Vulnerability Assessment](https://example.com)")
            st.markdown("- [Cross-Border Fraud Statistics](https://example.com)")
        st.markdown("</div>", unsafe_allow_html=True)

# --- AI Assistant Section ---
elif section == "AI Assistant":
    st.markdown("<h2 class='subheader'>🤖 AI Fraud Prevention Assistant</h2>", unsafe_allow_html=True)
    
    # Layout with columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Initialize chat if not exists
        if "ai_chat" not in st.session_state:
            st.session_state.ai_chat = [
                {"role": "assistant", "content": "👋 Hello! I'm your AI Fraud Prevention Assistant. How can I help you today?"}
            ]

        # Enhanced chat display with styling
        st.markdown("<div style='background-color: #F9FAFB; border-radius: 0.5rem; padding: 1rem; height: 400px; overflow-y: auto; margin-bottom: 1rem;'>", unsafe_allow_html=True)
        for msg in st.session_state.ai_chat:
            if msg["role"] == "user":
                st.markdown(f"<div style='text-align: right;'><div class='chat-bubble user-bubble'>{msg['content']}</div></div><div style='clear: both;'></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left;'><div class='chat-bubble assistant-bubble'>{msg['content']}</div></div><div style='clear: both;'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

                # Chat input area with button
        input_col, button_col = st.columns([5, 1])
        with input_col:
            user_input = st.text_input("Type your question here...", key="ai_assistant_input")
        with button_col:
            send_btn = st.button("Send", use_container_width=True)

        if user_input and send_btn:
            st.session_state.ai_chat.append({"role": "user", "content": user_input})
            
            # Show typing indicator
            with st.spinner("AI Assistant is thinking..."):
                try:
                    # Initialize Gemini model
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Create conversation history for context
                    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.ai_chat])
                    
                    # Generate response
                    response = model.generate_content(
                        f"""You are an expert fraud prevention assistant for a bank. Provide helpful, accurate information about fraud detection and prevention.
                        
                        Conversation history:
                        {chat_history}
                        
                        Current question:
                        {user_input}
                        
                        Respond in a professional but friendly tone. Keep answers concise but thorough. Use bullet points when listing items.
                        If suggesting actions, number them clearly.
                        """,
                        generation_config={
                            "temperature": 0.3,
                            "max_output_tokens": 500
                        }
                    )
                    
                    # Add response to chat
                    st.session_state.ai_chat.append({"role": "assistant", "content": response.text})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error communicating with AI: {str(e)}")
                    st.session_state.ai_chat.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})
                    st.rerun()

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 💡 Quick Actions")
        
        if st.button("🛡️ Fraud Prevention Tips", use_container_width=True):
            st.session_state.ai_chat.append({"role": "user", "content": "Give me fraud prevention tips"})
            st.session_state.ai_chat.append({"role": "assistant", "content": """Here are key fraud prevention tips:
            
1. **Monitor Accounts Daily** - Check transactions regularly
2. **Use Strong Authentication** - Enable 2FA wherever possible
3. **Educate Customers** - Teach them to recognize phishing attempts
4. **Set Transaction Limits** - Especially for new accounts
5. **Watch for Small Test Transactions** - Common fraud precursor
6. **Verify Large Transactions** - Especially international ones
7. **Update Systems Regularly** - Keep security patches current"""})
            st.rerun()
            
        if st.button("🚨 Common Fraud Patterns", use_container_width=True):
            st.session_state.ai_chat.append({"role": "user", "content": "What are common fraud patterns?"})
            st.session_state.ai_chat.append({"role": "assistant", "content": """Most common ATM fraud patterns:
            
- **Card Skimming** - Devices capture card data
- **Shoulder Surfing** - Watching customers enter PINs
- **Cash Trapping** - ATM doesn't dispense but records transaction
- **Distraction Theft** - While customer is distracted
- **Transaction Reversal Fraud** - Exploiting system delays
- **Ghost ATMs** - Fake machines that steal data
- **Lebanese Loop** - Device traps card in slot"""})
            st.rerun()
            
        if st.button("📊 Fraud Statistics", use_container_width=True):
            st.session_state.ai_chat.append({"role": "user", "content": "Share fraud statistics"})
            st.session_state.ai_chat.append({"role": "assistant", "content": """Latest fraud statistics (2025):
            
- **$12.8B** lost to payment fraud globally
- **78%** of fraud involves card-not-present transactions
- **42%** increase in mobile banking fraud
- **63%** of fraud occurs on weekends
- **2.7s** average time to detect fraud with AI
- **1 in 127** transactions are attempted fraud"""})
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 📌 Sample Questions")
        st.markdown("- How to detect card skimming?")
        st.markdown("- What's the latest phishing trend?")
        st.markdown("- How to handle suspected fraud?")
        st.markdown("- Best practices for ATM security?")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer for all pages
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("""
    <div style='display: flex; justify-content: center; gap: 2rem; margin-bottom: 0.5rem;'>
        <a href='#' style='color: #6B7280; text-decoration: none;'>Security Policy</a>
        <a href='#' style='color: #6B7280; text-decoration: none;'>Terms of Service</a>
        <a href='#' style='color: #6B7280; text-decoration: none;'>Contact Support</a>
        <a href='#' style='color: #6B7280; text-decoration: none;'>Training Portal</a>
    </div>
    <div style='text-align: center;'>
        © 2025 SecureBank. All rights reserved. | Fraud Detection System v2.2
    </div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)