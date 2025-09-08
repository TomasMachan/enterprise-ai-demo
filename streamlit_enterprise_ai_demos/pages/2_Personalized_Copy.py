import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap

st.title("✉️ Personalized Marketing Copy Generator")
st.caption("Generate persona-specific outreach emails with subject, body, and a single CTA.")

@st.cache_resource
def load_model():
    try:
        model_name = "google/flan-t5-base"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
        return tok, mdl, True
    except Exception as e:
        st.error(f"Failed to load AI model: {e}")
        st.info("Using fallback text generation...")
        return None, None, False

tokenizer, model, model_loaded = load_model()

# Show model status and generation options
if model_loaded:
    st.success("🤖 AI Model loaded successfully")
    use_ai = st.checkbox("Use AI generation (recommended)", value=True)
    if not use_ai:
        st.info("📝 Using template-based generation")
else:
    st.info("📝 Using template-based generation - AI model not available")
    use_ai = False

product = st.text_input("Product/Service", "cloud backup solution")
persona = st.selectbox("Persona", ["CFO", "IT Manager", "Small Business Owner"])
tone = st.selectbox("Tone", ["professional", "friendly", "bold"])
cta = st.selectbox("CTA", ["Book a demo", "Start free trial", "Get pricing"])
variants = st.slider("Number of variants", 1, 3, 3)

def generate_fallback_copy(product, persona, tone, cta):
    """Fallback text generation when AI model is not available"""
    
    # Handle modified personas (e.g., "CFO who is budget-conscious")
    base_persona = persona.split(" who is")[0].split(" at a")[0]
    
    templates = {
        "CFO": {
            "professional": f"Subject: Reduce Operational Costs by 30% with {product.title()}\n\nDear CFO,\n\nAs a financial leader, you're constantly seeking ways to optimize costs while maintaining quality. Our {product} has helped finance teams like yours achieve significant cost reductions through improved efficiency and streamlined operations.\n\nKey financial benefits:\n• Average 30% reduction in operational costs\n• Improved ROI within 6 months\n• Enhanced data security and compliance\n• Reduced manual processes and errors\n\nI'd love to show you how [Company Name] can help your organization save money while improving performance.\n\n{cta} to schedule a brief 15-minute call where I can share specific cost-saving strategies relevant to your industry.\n\nBest regards,\n[Your Name]\nSenior Sales Executive",
            
            "friendly": f"Subject: Quick Question About Your {product.title()} Costs\n\nHi there,\n\nI hope this email finds you well! I'm reaching out because I work with CFOs who are always looking for smart ways to reduce costs without sacrificing quality.\n\nOur {product} has been a game-changer for finance teams - helping them cut operational expenses by an average of 30% while actually improving their processes.\n\nI thought you might be interested in hearing about some of the cost-saving strategies we've implemented for companies similar to yours.\n\nWould you be open to a quick 15-minute conversation? {cta} and I'll share some insights that might be valuable for your team.\n\nThanks for your time!\n[Your Name]",
            
            "bold": f"Subject: Your Current {product.title()} is Costing You 40% More Than It Should\n\nDear CFO,\n\nHere's an uncomfortable truth: most companies are overspending on {product} by 40% due to inefficient solutions and poor optimization.\n\nOur {product} delivers immediate results:\n• 30% cost reduction from day one\n• Enterprise-grade security and compliance\n• Zero downtime implementation\n• Measurable ROI within 90 days\n\nDon't let another quarter pass with suboptimal performance eating into your budget.\n\n{cta} now and see exactly how much you could be saving.\n\nReady to stop wasting money?\n[Your Name]\nVP of Sales"
        },
        "IT Manager": {
            "professional": f"Subject: Streamline Your IT Infrastructure with {product.title()}\n\nDear IT Manager,\n\nManaging complex IT environments requires solutions that are both powerful and reliable. Our {product} is specifically designed for IT professionals who need seamless integration, advanced security, and 24/7 reliability.\n\nTechnical advantages:\n• Seamless integration with existing systems\n• Advanced security features and compliance\n• 24/7 monitoring and proactive support\n• Scalable architecture that grows with your needs\n\nI'd like to discuss how our {product} can enhance your IT operations and reduce your team's workload.\n\n{cta} to schedule a technical demo where we can explore integration possibilities and answer your specific questions.\n\nBest regards,\n[Your Name]\nTechnical Sales Engineer",
            
            "friendly": f"Subject: Hey IT Pro! Quick Question About {product.title()}\n\nHi,\n\nI know you're probably juggling a million things right now (aren't we all in IT?), but I thought you'd be interested in our {product}.\n\nIt's designed specifically for IT teams like yours - easy to deploy, integrates with your existing stack, and actually makes your job easier (I promise!).\n\nWe've helped IT managers reduce their daily firefighting by 50% while improving system reliability.\n\nWant to see how it works? {cta} and I'll show you a quick demo that won't waste your time.\n\nCheers,\n[Your Name]\nSolutions Architect",
            
            "bold": f"Subject: Stop Fighting with Outdated {product.title()} Systems\n\nDear IT Manager,\n\nYou're tired of band-aid solutions and constant troubleshooting. Your current {product} is holding your team back from focusing on strategic initiatives.\n\nOur {product} is built for modern IT teams who demand better:\n• Lightning-fast deployment (up and running in hours, not weeks)\n• Zero learning curve for your team\n• Enterprise-grade security and compliance\n• 99.9% uptime guarantee with proactive monitoring\n\n{cta} and see why forward-thinking IT teams are making the switch.\n\nTime to stop fighting your tools and start winning with them?\n[Your Name]\nVP of Engineering"
        },
        "Small Business Owner": {
            "professional": f"Subject: Scale Your Business with {product.title()}\n\nDear Business Owner,\n\nGrowing a business requires solutions that deliver enterprise-level results without enterprise-level complexity. Our {product} is designed specifically for small and medium businesses that need to compete with larger organizations.\n\nBusiness benefits:\n• Affordable pricing with transparent, no-hidden-fee structure\n• Easy setup and management (no technical expertise required)\n• Scalable solution that grows with your business\n• Dedicated support team that understands small business challenges\n\nI'd love to show you how other businesses like yours have used our {product} to increase efficiency and drive growth.\n\n{cta} to schedule a brief consultation where we can discuss your specific business goals and how we can help you achieve them.\n\nBest regards,\n[Your Name]\nBusiness Development Manager",
            
            "friendly": f"Subject: Quick Question About Growing Your Business\n\nHi there,\n\nI hope business is going well! I wanted to reach out because I work with small business owners like you who are looking to grow and improve their operations.\n\nOur {product} has helped hundreds of small businesses streamline their processes, save time, and increase their revenue - all without breaking the bank.\n\nI thought you might be interested in hearing about some of the success stories from businesses similar to yours.\n\nInterested in learning more? {cta} and let's chat about how we can help your business succeed and grow.\n\nLooking forward to hearing from you!\n[Your Name]\nSmall Business Specialist",
            
            "bold": f"Subject: Your Competitors Are Using {product.title()} - Are You?\n\nDear Business Owner,\n\nEvery day you wait is money lost. Your competitors are already using solutions like our {product} to get ahead, while you're stuck with outdated processes.\n\nHere's what you're missing:\n• 25% increase in operational efficiency\n• Reduced costs and improved profit margins\n• Better customer satisfaction and retention\n• Competitive advantage in your market\n\nDon't get left behind while your competitors pull ahead.\n\n{cta} now and see how you can start winning today.\n\nReady to compete and win?\n[Your Name]\nGrowth Specialist"
        }
    }
    
    return templates.get(base_persona, {}).get(tone, f"Subject: {product.title()} Solution for {persona}\n\nDear {persona},\n\nI wanted to introduce you to our {product} solution. It's designed to help professionals like you achieve better results and improve your operations.\n\nKey benefits:\n• Improved efficiency and productivity\n• Cost-effective solution\n• Easy to implement and use\n• Dedicated support team\n\n{cta} to learn more about how we can help you succeed.\n\nBest regards,\n[Your Name]\nSales Team")

def generate_copy(product, persona, tone, cta):
    if model_loaded and tokenizer and model and use_ai:
        try:
            # Improved prompt for better email generation
            prompt = f"""Write a professional B2B sales email with:

Subject: [compelling subject line]

Body: [120-180 words, {tone} tone]

Product: {product}
Target: {persona}
Call-to-action: {cta}

Format:
Subject: [subject line]
[email body with clear value proposition and single CTA]"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=300, 
                num_beams=4, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                early_stopping=True
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the generated text
            if "Subject:" in text:
                return text
            else:
                # If AI didn't follow format, use fallback
                return generate_fallback_copy(product, persona, tone, cta)
                
        except Exception as e:
            st.warning(f"AI generation failed: {e}. Using fallback.")
            return generate_fallback_copy(product, persona, tone, cta)
    else:
        return generate_fallback_copy(product, persona, tone, cta)

if st.button("Generate"):
    for i in range(variants):
        if i == 1:
            p = persona + " who is budget-conscious"
        elif i == 2:
            p = persona + " at a growing company"
        else:
            p = persona
        
        st.markdown(f"### Variant {i+1}")
        
        # Generate the email
        email_content = generate_copy(product, p, tone, cta)
        
        # Format the email nicely
        if "Subject:" in email_content:
            parts = email_content.split("Subject:", 1)
            if len(parts) == 2:
                subject = parts[1].split("\n")[0].strip()
                body = parts[1].split("\n", 1)[1].strip() if len(parts[1].split("\n", 1)) > 1 else ""
                
                st.markdown(f"**Subject:** {subject}")
                st.markdown("**Body:**")
                st.write(textwrap.fill(body, width=80))
            else:
                st.write(textwrap.fill(email_content, width=80))
        else:
            st.write(textwrap.fill(email_content, width=80))
        
        st.markdown("---")
