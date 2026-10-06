# AI-WAIFU-STAGE-1
An offline first ai-self trainend ,Alicia Elswarth is an experimental offline-first personal AI project built with Python. Stage 1 focuses on creating a lightweight AI that can learn from user-provided datasets and store its knowledge locally on the device.

import os
import json
import re
import math
import operator

# ==========================================
# 1. TOKENIZER & SIMILARITY ENGINE 
# ==========================================
class AliciaTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inv_vocab = {0: "<PAD>", 1: "<UNK>"}
        
    def fit_on_text(self, text):
        tokens = self.tokenize(text)
        for t in tokens:
            if t not in self.vocab:
                idx = len(self.vocab)
                self.vocab[t] = idx
                self.inv_vocab[idx] = t

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r'[\w\u0980-\u09FF]+|[^\w\s]', text)
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(t, 1) for t in tokens]


class AliciaSimilarity:
    @staticmethod
    def jaccard_similarity(str1, str2):
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    @staticmethod
    def char_ngram_similarity(str1, str2, n=2):
        def get_ngrams(s, n):
            s = f"_{s.lower()}_"
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        ng1 = get_ngrams(str1, n)
        ng2 = get_ngrams(str2, n)
        if not ng1 or not ng2:
            return 0.0
        intersection = ng1.intersection(ng2)
        union = ng1.union(ng2)
        return len(intersection) / len(union)

    @classmethod
    def compute_score(cls, query, target):
        jaccard = cls.jaccard_similarity(query, target)
        ngram = cls.char_ngram_similarity(query, target)
        return (jaccard * 0.6) + (ngram * 0.4)


# ==========================================
# 2. MEMORY MANAGEMENT
# ==========================================
class AliciaMemoryManager:
    def __init__(self, storage_file="alicia_stage2_memory.json"):
        self.storage_file = storage_file
        self.data = {
            "long_term": {},      
            "personal_facts": {}, 
            "knowledge": {}       
        }
        self.load_persistence()

    def load_persistence(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"[Memory] Loaded successfully from backend storage.")
            except Exception as e:
                print(f"[Memory] Error loading storage: {e}")

    def save_persistence(self):
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Memory] Error saving storage: {e}")

    def set_long_term(self, query, output, confidence=1.0):
        key = query.strip().lower()
        self.data["long_term"][key] = {
            "output": output.strip(),
            "confidence": confidence,
            "type": "learned_response"
        }
        self.save_persistence()

    def import_external_memory(self, file_path):
        """Open external long-term memory file and train the system with its data"""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    ext_data = json.load(f)
                    target_dict = ext_data.get("long_term", ext_data)
                    count = 0
                    for k, v in target_dict.items():
                        if isinstance(v, dict) and "output" in v:
                            self.data["long_term"][k.strip().lower()] = v
                        else:
                            self.data["long_term"][k.strip().lower()] = {
                                "output": str(v).strip(),
                                "confidence": 1.0,
                                "type": "imported_response"
                            }
                        count += 1
                    self.save_persistence()
                    return f"Successfully imported {count} items from {file_path} into long-term memory!"
            except Exception as e:
                return f"Error importing memory file: {e}"
        return f"File '{file_path}' not found."

    def get_semantic_match(self, query):
        best_match = None
        highest_score = 0.0
        query_clean = query.strip().lower()

        if query_clean in self.data["long_term"]:
            item = self.data["long_term"][query_clean]
            return item["output"], item["confidence"], 1.0

        for stored_q, item in self.data["long_term"].items():
            score = AliciaSimilarity.compute_score(query_clean, stored_q)
            if score > highest_score:
                highest_score = score
                best_match = item["output"]

        return best_match, highest_score, highest_score

    def set_personal_fact(self, attr, val):
        self.data["personal_facts"][attr.lower()] = val.strip()
        self.save_persistence()

    def get_personal_fact(self, attr):
        return self.data["personal_facts"].get(attr.lower())

    def set_knowledge(self, topic, explanation):
        self.data["knowledge"][topic.strip().lower()] = explanation.strip()
        self.save_persistence()

    def get_knowledge(self, topic):
        topic_clean = topic.strip().lower()
        for k, v in self.data["knowledge"].items():
            if k in topic_clean or topic_clean in k:
                return v
        return None


class AliciaShortTermContext:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []
        self.last_subject = None

    def add(self, role, text):
        self.history.append({"role": role, "text": text})
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        text_lower = text.lower()
        if "python" in text_lower:
            self.last_subject = "Python"
        elif "manga" in text_lower:
            self.last_subject = "manga"

    def resolve_context(self, text):
        if self.last_subject and any(w in text.lower() for w in ["it", "this", "eta", "eita"]):
            return f"{text} (about {self.last_subject})"
        return text


# ==========================================
# 3. TOOLS & REASONING ENGINE
# ==========================================
class AliciaTools:
    @staticmethod
    def evaluate_math(expression):
        try:
            expr = expression.lower().replace("×", "*").replace("x", "*").replace("÷", "/")
            expr = expr.replace("^", "**")
            allowed_chars = "0123456789+-*/().* "
            clean_expr = "".join([c for c in expr if c in allowed_chars])
            if clean_expr:
                result = eval(clean_expr)
                return str(result)
        except Exception:
            pass
        return None


class AliciaReasoningEngine:
    @staticmethod
    def evaluate_logic(query):
        query_lower = query.lower()
        if ">" in query_lower:
            parts = [p.strip() for p in query_lower.split(">")]
            if len(parts) == 2:
                return f"Inferred relationship: {parts[0].upper()} is greater than {parts[1].upper()}."
        return None


# ==========================================
# 4. INTENT CLASSIFIER
# ==========================================
class AliciaIntentClassifier:
    @staticmethod
    def classify(text):
        t = text.strip().lower()
        if t.startswith("/alicia"):
            return "COMMAND"
        if any(g in t for g in ["hi", "hii", "hello", "kemon acho", "greetings"]):
            return "GREETING"
        if any(m in t for m in ["+", "-", "*", "/", "×", "÷", "^", "calculate"]):
            if any(char.isdigit() for char in t):
                return "MATH"
        if "my name is" in t or "i like" in t or "i am learning" in t:
            return "PERSONAL_FACT"
        if "what is my name" in t or "what do i like" in t or "what am i learning" in t:
            return "MEMORY_QUERY"
        if ">" in t or "if" in t:
            return "LOGIC"
        if "what is" in t or "define" in t or "who is" in t:
            return "KNOWLEDGE_QUERY"
        return "CONVERSATION"


# ==========================================
# 5. MASTER CONTROLLER / ALICIA SML STAGE 2
# ==========================================
class Alicia3Stage2:
    def __init__(self):
        self.tokenizer = AliciaTokenizer()
        self.memory = AliciaMemoryManager()
        self.context = AliciaShortTermContext()
        self.last_debug_info = {}
        self.pending_learning_query = None  # Holds temporary query for auto-learning

    def process(self, user_input):
        raw_input = user_input.strip()
        if not raw_input:
            return "Please provide a valid input."

        # If an answer was awaited for a previous unknown query
        if self.pending_learning_query:
            ans = raw_input
            q = self.pending_learning_query
            self.pending_learning_query = None
            self.memory.set_long_term(q, ans, confidence=1.0)
            return f"Thank you! I learned that '{q}' means '{ans}'."

        resolved_input = self.context.resolve_context(raw_input)
        self.context.add("User", raw_input)

        intent = AliciaIntentClassifier.classify(resolved_input)
        source = "Unknown"
        confidence = 0.0
        reasoning = "No matching rule found."
        response = None

        if intent == "COMMAND":
            return self.handle_command(raw_input)

        math_ans = AliciaTools.evaluate_math(resolved_input)
        if math_ans:
            intent = "MATH"
            source = "Calculator Tool"
            confidence = 1.0
            reasoning = "Detected mathematical expression."
            response = f"{raw_input} = {math_ans}"

        elif intent == "PERSONAL_FACT":
            response = self.handle_personal_fact_storage(resolved_input)
            source = "Personal Memory Storage"
            confidence = 1.0
            reasoning = "Extracted and stored explicit personal attribute."

        elif intent == "MEMORY_QUERY":
            response = self.handle_personal_fact_query(resolved_input)
            if response:
                source = "Personal Memory"
                confidence = 0.96
                reasoning = "Retrieved stored user attribute."

        if not response:
            mem_out, conf, sim_score = self.memory.get_semantic_match(resolved_input)
            if mem_out and conf >= 0.55:
                response = mem_out
                confidence = conf
                source = "Semantic Memory"
                reasoning = f"Matched via similarity engine (Score: {conf:.2f})."

        if not response and intent == "KNOWLEDGE_QUERY":
            knowledge_ans = self.memory.get_knowledge(resolved_input)
            if knowledge_ans:
                response = knowledge_ans
                source = "Knowledge Database"
                confidence = 0.90
                reasoning = "Retrieved verified fact from local knowledge store."

        if not response:
            logic_res = AliciaReasoningEngine.evaluate_logic(resolved_input)
            if logic_res:
                response = logic_res
                source = "Reasoning Engine"
                confidence = 1.0
                reasoning = "Evaluated deterministic logic pattern."

        # Fallback and auto-learning trigger
        if not response or confidence < 0.55:
            self.pending_learning_query = raw_input  # Stored query to learn from user's next message
            source = "Unknown"
            confidence = 0.0
            reasoning = "Confidence below threshold. Triggered auto-learning prompt."
            response = "I don't know this yet. What should be the answer?"

        self.last_debug_info = {
            "intent": intent,
            "source": source,
            "matched": resolved_input,
            "confidence": confidence,
            "reasoning": reasoning
        }

        self.context.add("Alicia", response)
        return response

    def handle_personal_fact_storage(self, text):
        t_low = text.lower()
        if "my name is" in t_low:
            name = text.split("is")[-1].strip()
            self.memory.set_personal_fact("name", name)
            return f"Nice to meet you, {name}."
        elif "i like" in t_low:
            like_val = text.split("like")[-1].strip()
            self.memory.set_personal_fact("likes", like_val)
            return f"Got it! I will remember that you like {like_val}."
        elif "i am learning" in t_low:
            learning_val = text.split("learning")[-1].strip()
            self.memory.set_personal_fact("learning", learning_val)
            return f"That's awesome! Learning {learning_val} is a great choice."
        return "I saved your personal information."

    def handle_personal_fact_query(self, text):
        t_low = text.lower()
        if "what is my name" in t_low:
            val = self.memory.get_personal_fact("name")
            return f"Your name is {val}." if val else "I don't know your name yet."
        elif "what do i like" in t_low:
            val = self.memory.get_personal_fact("likes")
            return f"You like {val}." if val else "I don't know what you like yet."
        elif "what am i learning" in t_low:
            val = self.memory.get_personal_fact("learning")
            return f"You are learning {val}." if val else "I don't know what you are learning yet."
        return None

    def handle_command(self, cmd):
        c = cmd.lower().strip()
        if c.startswith("/alicia import"):
            parts = cmd.split(" ", 2)
            if len(parts) > 2:
                file_path = parts[2].strip()
                return self.memory.import_external_memory(file_path)
            return "Please provide a file path. Example: /alicia import dataset.json"
        elif c == "/alicia memory":
            return f"[Memory Stats] Long-term entries: {len(self.memory.data['long_term'])}, Personal facts: {len(self.memory.data['personal_facts'])}"
        elif c == "/alicia why":
            debug = self.last_debug_info
            return (f"\n--- Alicia Debug Info (/alicia why) ---\n"
                    f"Intent: {debug.get('intent')}\n"
                    f"Source: {debug.get('source')}\n"
                    f"Matched Query: {debug.get('matched')}\n"
                    f"Confidence: {debug.get('confidence'):.2f}\n"
                    f"Reasoning: {debug.get('reasoning')}\n---------------------------------------")
        elif c == "/alicia stats":
            return f"[System Stats] Stage: 2, Vocabulary Size: {len(self.tokenizer.vocab)}, Storage File: {self.memory.storage_file}"
        elif c == "/alicia help":
            return ("Commands available:\n"
                    " - /alicia memory\n"
                    " - /alicia why\n"
                    " - /alicia stats\n"
                    " - /alicia import <filename> (To train from external long-term memory file)\n"
                    " - /alicia help")
        return "Unknown command. Type '/alicia help' for assistance."


# ==========================================
# 6. TERMINAL RUNTIME INTERFACE
# ==========================================
if __name__ == "__main__":
    alicia = Alicia3Stage2()
    print("\n=======================================================")
    print("      Alicia Elswarth-3 SML — STAGE 2 (FROM-PYDROID3)        ")
    print("  Your honest trained AI is ready for your questions.(>•<) ")
    print("=======================================================")
    print("Type 'exit' to quit. Type '/alicia help' for commands.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Alicia: Goodbye! All memory safely persisted.")
                break
            
            response = alicia.process(user_input)
            print(f"Alicia: {response}\n")
        except KeyboardInterrupt:
            print("\nAlicia: Shutting down safely.")
            break
