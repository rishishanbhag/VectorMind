import json
import logging
import os
import pickle
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from neo4j import GraphDatabase
from PyPDF2 import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

PARENT_SIZE = int(os.getenv("PARENT_SIZE", "2000"))
PARENT_OVERLAP = int(os.getenv("PARENT_OVERLAP", "200"))
CHILD_SIZE = int(os.getenv("CHILD_SIZE", "400"))
CHILD_OVERLAP = int(os.getenv("CHILD_OVERLAP", "50"))
TOP_K_CHILDREN = int(os.getenv("TOP_K_CHILDREN", "8"))
OVERFETCH_K = int(os.getenv("OVERFETCH_K", "20"))
MAX_PARENTS = int(os.getenv("MAX_PARENTS", "6"))
MAX_L2_DISTANCE = float(os.getenv("MAX_L2_DISTANCE", "2.0"))
INDEX_FILE = os.getenv("INDEX_FILE", "vectorstore.pkl")
SUMMARY_CHAR_BUDGET = int(os.getenv("SUMMARY_CHAR_BUDGET", "12000"))
SYNTHESIS_TOP_M = int(os.getenv("SYNTHESIS_TOP_M", "12"))
CONTEXT_CHAR_BUDGET = int(os.getenv("CONTEXT_CHAR_BUDGET", "14000"))
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

SYNTHESIS_PATTERNS = re.compile(
    r"\b(each|every|all)\b.*\b(document|documents|pdf|pdfs|file|files)\b"
    r"|\b(compare|across all|overview of all|summarize each|from every)\b"
    r"|\beach and every\b",
    re.IGNORECASE,
)
# ponytail: keyword heuristic for cross-doc; upgrade path = small LLM classifier
CROSS_DOC_PATTERNS = re.compile(
    r"\b(more than one|across|overlap|overlapping|common|shared)\b.*\b(concept|concepts|topic|topics|entity|entities)\b"
    r"|\b(concept|concepts)\b.*\b(more than one|multiple|across|common|shared)\b"
    r"|\b(complement|differ|difference|differences|compare|comparison)\b.*\b(document|documents|pdf|pdfs)\b"
    r"|\b(document|documents|pdf|pdfs)\b.*\b(complement|differ|overlap|common concepts)\b"
    r"|\bappears in more than one\b"
    r"|\bin more than one document\b",
    re.IGNORECASE,
)
CROSS_DOC_CONCEPT_LIMIT = int(os.getenv("CROSS_DOC_CONCEPT_LIMIT", "20"))
CROSS_DOC_CONTEXT_BUDGET = int(os.getenv("CROSS_DOC_CONTEXT_BUDGET", "18000"))


class PDFChatbot:
    def __init__(self):
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        self.vectorstore: Optional[FAISS] = None
        # parent_id -> {text, doc_id, doc_name}
        self.parents: Dict[str, Dict[str, str]] = {}
        self.child_to_parent: Dict[str, str] = {}
        # doc_id -> registry entry
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chat_history: List[Tuple[str, str]] = []
        self.ready = False
        self._driver = None
        self._embeddings = None
        self._reranker = None

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embeddings

    def _get_reranker(self): # this is a function to get the reranker for the children
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        return self._reranker

    def _get_llm(self, temperature: float = 0.2):
        return ChatAnthropic(
            model=self.claude_model,
            temperature=temperature,
            anthropic_api_key=self.anthropic_api_key,
        )

    def _get_driver(self):
        if not self.neo4j_uri or not self.neo4j_password:
            raise ValueError(
                "Neo4j Aura credentials missing. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
            )
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            self._driver.verify_connectivity()
        return self._driver

    def _extract_pdf_text(self, pdf) -> str:
        try:
            return "".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""

    def get_pdf_text(self, pdf_docs) -> str:
        return "".join(self._extract_pdf_text(pdf) for pdf in pdf_docs)

    def get_pdf_text_from_paths(self, pdf_paths: List[str]) -> str:
        texts = []
        for pdf_path in pdf_paths:
            try:
                with open(pdf_path, "rb") as file:
                    texts.append(self._extract_pdf_text(file))
            except Exception as e:
                logger.error(f"Error reading PDF {pdf_path}: {e}")
        return "".join(texts)

    def get_parent_child_chunks(
        self,
        raw_text: str,
        doc_id: str,
        doc_name: str,
    ) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]]]:
        """Split into parents/children tagged with doc_id and doc_name."""
        parent_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=PARENT_SIZE,
            chunk_overlap=PARENT_OVERLAP,
            length_function=len,
        )
        child_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=CHILD_SIZE,
            chunk_overlap=CHILD_OVERLAP,
            length_function=len,
        )

        parents: Dict[str, Dict[str, str]] = {}
        children: List[Dict[str, str]] = []

        for parent_text in parent_splitter.split_text(raw_text):
            parent_id = f"p_{uuid.uuid4().hex[:10]}"
            parents[parent_id] = {
                "text": parent_text,
                "doc_id": doc_id,
                "doc_name": doc_name,
            }
            for child_text in child_splitter.split_text(parent_text):
                child_id = f"c_{uuid.uuid4().hex[:10]}"
                children.append(
                    {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "text": child_text,
                    }
                )
        return parents, children

    def create_vectorstore(self, children: List[Dict[str, str]]) -> FAISS:
        texts = [c["text"] for c in children]
        metadatas = [
            {
                "child_id": c["child_id"],
                "parent_id": c["parent_id"],
                "doc_id": c["doc_id"],
                "doc_name": c["doc_name"],
            }
            for c in children
        ]
        vectorstore = FAISS.from_texts(
            texts, embedding=self._get_embeddings(), metadatas=metadatas
        )
        self.vectorstore = vectorstore
        self.child_to_parent = {c["child_id"]: c["parent_id"] for c in children}
        return vectorstore

    def save_index(self):
        if self.vectorstore is None:
            raise ValueError("No vectorstore to save")
        payload = {
            "vectorstore": self.vectorstore,
            "parents": self.parents,
            "child_to_parent": self.child_to_parent,
            "documents": self.documents,
        }
        with open(INDEX_FILE, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Index saved successfully")

    def load_index(self) -> bool:
        try:
            with open(INDEX_FILE, "rb") as f:
                payload = pickle.load(f)
            if not (isinstance(payload, dict) and "vectorstore" in payload):
                logger.warning("Legacy vectorstore format found; please re-process documents.")
                return False
            self.vectorstore = payload["vectorstore"]
            raw_parents = payload.get("parents", {})
            # Normalize legacy parents (str values) if present
            self.parents = {}
            for pid, val in raw_parents.items():
                if isinstance(val, dict):
                    self.parents[pid] = val
                else:
                    self.parents[pid] = {
                        "text": val,
                        "doc_id": "legacy",
                        "doc_name": "legacy",
                    }
            self.child_to_parent = payload.get("child_to_parent", {})
            self.documents = payload.get("documents", {})
            self.ready = True
            logger.info("Index loaded successfully")
            return True
        except FileNotFoundError:
            logger.warning("No index found. Please process documents first.")
            return False
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def clear_knowledge_graph(self):
        driver = self._get_driver()
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j graph cleared")

    def _ensure_graph_constraints(self, session):
        session.run(
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
        )

    @staticmethod

    def _llm_text(content) -> str: # this is a function to convert the content by claude to a string 
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content or "")

    @staticmethod
    def _parse_json_object(raw: str) -> Optional[dict]: # this is a function to parse the json object from the content by claude
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _sample_representative_parents( # this is a function to sample the representative parents from the parent texts 
        parent_texts: List[str], budget: int
    ) -> str:
        """Pick beginning/middle/end plus evenly spaced parents to fit budget."""
        if not parent_texts:
            return ""
        joined = "\n\n".join(parent_texts)
        if len(joined) <= budget:
            return joined

        n = len(parent_texts)
        idxs = {0, n // 2, n - 1}
        # evenly sample more until we fill or run out
        step = max(1, n // 8)
        for i in range(0, n, step):
            idxs.add(i)
        ordered = [parent_texts[i] for i in sorted(idxs)]
        out, used = [], 0
        for t in ordered:
            if used + len(t) + 2 > budget:
                remain = budget - used - 2
                if remain > 200:
                    out.append(t[:remain])
                break
            out.append(t)
            used += len(t) + 2
        return "\n\n".join(out)

    #Sample parent text.
    # Ask Claude for JSON: {summary, topics, keywords}.
    # Parse it; on failure, fall back to the first 500 chars of the source.
    # Return (summary, topics, keywords) for the Document Registry.
    def summarize_document(   
        self, doc_name: str, parent_texts: List[str]
    ) -> Tuple[str, List[str], List[str]]:
        """
        Adaptive one-call summary from parent chunks.
        Returns (summary, topics, keywords).
        """
        source = self._sample_representative_parents(parent_texts, SUMMARY_CHAR_BUDGET)
        prompt = (
            "Summarize this document for retrieval routing.\n"
            "Reply with ONLY JSON (no markdown):\n"
            '{"summary":"2-4 sentence overview","topics":["..."],"keywords":["..."]}\n\n'
            f"Document name: {doc_name}\n\nText:\n{source}"
        )
        raw = self._llm_text(self._get_llm(temperature=0).invoke(prompt).content)
        data = self._parse_json_object(raw)
        if not data:
            logger.warning("Summary JSON parse failed for %s", doc_name)
            return source[:500], [], []
        summary = str(data.get("summary") or "")[:2000]
        topics = [str(t) for t in (data.get("topics") or []) if t][:12]
        keywords = [str(k) for k in (data.get("keywords") or []) if k][:20]
        return summary or source[:500], topics, keywords

    def _extract_triples(self, chunk_text: str) -> List[Dict[str, str]]:
        llm = self._get_llm(temperature=0)
        prompt = (
            "Extract entities and relationships from the text.\n"
            "Reply with ONLY a JSON object, no markdown, no prose.\n"
            'Schema: {"triples":[{"source":"...","relation":"...","target":"...","source_type":"...","target_type":"..."}]}\n'
            "Use short relation names in UPPER_SNAKE_CASE. If none, return {\"triples\":[]}.\n\n"
            f"Text:\n{chunk_text[:2500]}"
        )
        raw = self._llm_text(llm.invoke(prompt).content)
        data = self._parse_json_object(raw)
        if data is None:
            logger.warning(
                "Failed to parse entity JSON: %s", raw[:200].replace("\n", " ")
            )
            return []
        triples = data.get("triples", [])
        return triples if isinstance(triples, list) else []

    def build_knowledge_graph(
        self,
        doc_id: str,
        doc_name: str,
        parents: Dict[str, Dict[str, str]],
        summary: str = "",
    ) -> List[str]:
        """Write graph for one document. Returns entity names mentioned."""
        entities: Set[str] = set()
        driver = self._get_driver()
        with driver.session() as session:
            self._ensure_graph_constraints(session)
            session.run(
                "MERGE (d:Document {id: $id}) SET d.name = $name, d.summary = $summary",
                id=doc_id,
                name=doc_name,
                summary=summary,
            )

            for parent_id, meta in parents.items():
                parent_text = meta["text"] if isinstance(meta, dict) else meta
                session.run(
                    """
                    MERGE (c:Chunk {id: $cid})
                    SET c.text = $text, c.doc_id = $doc_id
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    cid=parent_id,
                    text=parent_text,
                    doc_id=doc_id,
                )

                triples = self._extract_triples(parent_text)
                for t in triples:
                    source = (t.get("source") or "").strip()
                    target = (t.get("target") or "").strip()
                    relation = (
                        (t.get("relation") or "RELATED_TO")
                        .strip()
                        .upper()
                        .replace(" ", "_")
                    )
                    if not source or not target:
                        continue
                    entities.add(source)
                    entities.add(target)
                    source_type = (t.get("source_type") or "Concept").strip()
                    target_type = (t.get("target_type") or "Concept").strip()
                    session.run(
                        """
                        MATCH (c:Chunk {id: $cid})
                        MERGE (s:Entity {name: $source})
                        SET s.type = $source_type
                        MERGE (t:Entity {name: $target})
                        SET t.type = $target_type
                        MERGE (c)-[:MENTIONS]->(s)
                        MERGE (c)-[:MENTIONS]->(t)
                        MERGE (s)-[r:REL]->(t)
                        SET r.type = $relation
                        """,
                        cid=parent_id,
                        source=source,
                        target=target,
                        source_type=source_type,
                        target_type=target_type,
                        relation=relation,
                    )
        logger.info(
            "Knowledge graph built for %s (%d parents, %d entities)",
            doc_name,
            len(parents),
            len(entities),
        )
        return sorted(entities)

    def get_graph_neighborhood(
        self, parent_ids: List[str], question: str
    ) -> List[str]:
        if not parent_ids:
            return []
        driver = self._get_driver()
        facts: List[str] = []
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.id IN $ids
                OPTIONAL MATCH (e)-[r:REL]-(other:Entity)
                RETURN DISTINCT e.name AS entity, e.type AS etype,
                       r.type AS rel, other.name AS other
                LIMIT 40
                """,
                ids=parent_ids,
            )
            for row in result:
                if row["rel"] and row["other"]:
                    facts.append(f"{row['entity']} -[{row['rel']}]-> {row['other']}")
                elif row["entity"]:
                    facts.append(f"{row['entity']} ({row['etype'] or 'Entity'})")

            q_lower = question.lower()
            ent_result = session.run(
                """
                MATCH (e:Entity)-[r:REL]-(other:Entity)
                RETURN e.name AS entity, r.type AS rel, other.name AS other
                LIMIT 80
                """
            )
            for row in ent_result:
                name = row["entity"] or ""
                if name and name.lower() in q_lower:
                    facts.append(f"{name} -[{row['rel']}]-> {row['other']}")

        seen, unique = set(), []
        for f in facts:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique[:30]

    def route_query(self, question: str) -> str:
        q = question or ""
        # ponytail: keyword heuristic; upgrade path = small LLM classifier if misroutes
        if CROSS_DOC_PATTERNS.search(q):
            return "cross_doc"
        if SYNTHESIS_PATTERNS.search(q):
            # Promote to cross_doc when asking concepts/compare across 2+ docs
            if len(self.documents) >= 2 and re.search(
                r"\b(concept|concepts|topic|topics|compare|differ|complement|overlap)\b",
                q,
                re.IGNORECASE,
            ):
                return "cross_doc"
            return "synthesis"
        return "lookup"

    def _registry_cross_doc_entities(
        self, min_docs: int = 2, limit: int = CROSS_DOC_CONCEPT_LIMIT
    ) -> List[Dict[str, Any]]:
        """In-memory fallback: terms appearing in 2+ document registries."""
        counts: Dict[str, Dict[str, Any]] = {}
        for doc_id, meta in self.documents.items():
            doc_name = meta.get("doc_name", doc_id)
            seen_in_doc: Set[str] = set()
            for field in ("entities", "topics", "keywords"):
                for term in meta.get(field) or []:
                    key = str(term).strip().lower()
                    if len(key) < 3 or key in seen_in_doc:
                        continue
                    seen_in_doc.add(key)
                    bucket = counts.setdefault(
                        key, {"name": str(term).strip(), "docs": set(), "doc_names": {}}
                    )
                    bucket["docs"].add(doc_id)
                    bucket["doc_names"][doc_id] = doc_name
        results = []
        for key, bucket in counts.items():
            if len(bucket["docs"]) >= min_docs:
                docs = sorted(bucket["docs"])
                results.append(
                    {
                        "name": bucket["name"],
                        "docs": docs,
                        "doc_names": [bucket["doc_names"][d] for d in docs],
                        "doc_count": len(docs),
                    }
                )
        results.sort(key=lambda x: (-x["doc_count"], x["name"].lower()))
        return results[:limit]

    def get_cross_doc_entities(
        self, min_docs: int = 2, limit: int = CROSS_DOC_CONCEPT_LIMIT
    ) -> List[Dict[str, Any]]:
        """Entities mentioned by chunks from >= min_docs distinct documents."""
        results: List[Dict[str, Any]] = []
        try:
            driver = self._get_driver()
            with driver.session() as session:
                rows = session.run(
                    """
                    MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
                    WHERE c.doc_id IS NOT NULL
                    WITH e, collect(DISTINCT c.doc_id) AS docs
                    WHERE size(docs) >= $min_docs
                    RETURN e.name AS name, docs, size(docs) AS doc_count
                    ORDER BY doc_count DESC, name
                    LIMIT $limit
                    """,
                    min_docs=min_docs,
                    limit=limit,
                )
                for row in rows:
                    docs = [d for d in (row["docs"] or []) if d in self.documents]
                    if len(docs) < min_docs:
                        continue
                    results.append(
                        {
                            "name": row["name"],
                            "docs": docs,
                            "doc_names": [
                                (self.documents.get(d) or {}).get("doc_name", d)
                                for d in docs
                            ],
                            "doc_count": len(docs),
                        }
                    )
        except Exception as e:
            logger.warning("Neo4j cross-doc entity query failed: %s", e)

        if results:
            return results[:limit]
        return self._registry_cross_doc_entities(min_docs=min_docs, limit=limit)

    def _neo4j_chunk_for_entity_doc(
        self, entity_name: str, doc_id: str
    ) -> Optional[str]:
        """Return a Chunk.id that mentions entity in doc, if any."""
        try:
            driver = self._get_driver()
            with driver.session() as session:
                row = session.run(
                    """
                    MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
                    WHERE toLower(e.name) = toLower($name) AND c.doc_id = $doc_id
                    RETURN c.id AS cid
                    LIMIT 1
                    """,
                    name=entity_name,
                    doc_id=doc_id,
                ).single()
                if row and row["cid"]:
                    return row["cid"]
        except Exception as e:
            logger.warning("Neo4j chunk lookup failed: %s", e)
        return None

    def _excerpt_for_concept_doc(
        self, concept: str, doc_id: str, question: str
    ) -> Optional[Dict[str, str]]:
        """One grounded parent excerpt for concept within a document."""
        chunk_id = self._neo4j_chunk_for_entity_doc(concept, doc_id)
        if chunk_id and chunk_id in self.parents:
            return self._parent_record(chunk_id)

        # Chunk ids are parent ids in our graph (we store parents as Chunk nodes)
        query = f"{concept} {question}".strip()
        hits = self._faiss_search(query, k=3, doc_ids=[doc_id])
        if hits:
            parent_id = hits[0][0].metadata.get("parent_id")
            if parent_id:
                return self._parent_record(parent_id)

        for pid, pmeta in self.parents.items():
            if isinstance(pmeta, dict) and pmeta.get("doc_id") == doc_id:
                text = (pmeta.get("text") or "").lower()
                if concept.lower() in text:
                    return self._parent_record(pid)
        for pid, pmeta in self.parents.items():
            if isinstance(pmeta, dict) and pmeta.get("doc_id") == doc_id:
                return self._parent_record(pid)
        return None

    def cross_doc_retrieve(self, question: str) -> Dict[str, Any]:
        shared = self.get_cross_doc_entities(
            min_docs=2, limit=CROSS_DOC_CONCEPT_LIMIT
        )
        candidate_docs: List[str] = []
        for c in shared:
            for d in c.get("docs") or []:
                if d not in candidate_docs:
                    candidate_docs.append(d)
        if not candidate_docs:
            candidate_docs = list(self.documents.keys())

        summaries = []
        for doc_id in candidate_docs:
            meta = self.documents.get(doc_id) or {}
            summaries.append(
                {
                    "doc_id": doc_id,
                    "doc_name": meta.get("doc_name", doc_id),
                    "summary": meta.get("summary", ""),
                }
            )

        parents: List[Dict[str, str]] = []
        seen_parent_ids: Set[str] = set()
        for concept in shared:
            name = concept["name"]
            for doc_id in concept.get("docs") or []:
                rec = self._excerpt_for_concept_doc(name, doc_id, question)
                if not rec:
                    continue
                pid = rec["id"]
                # Tag concept on the excerpt for the prompt
                tagged = dict(rec)
                tagged["concept"] = name
                if pid not in seen_parent_ids:
                    seen_parent_ids.add(pid)
                    parents.append(tagged)
                else:
                    # already have parent; still useful as concept note via graph_facts
                    pass

        parents = self._budget_parents(parents, CROSS_DOC_CONTEXT_BUDGET)

        concept_lines = [
            f"{c['name']} — docs: {', '.join(c.get('doc_names') or c.get('docs') or [])}"
            for c in shared
        ]
        return {
            "mode": "cross_doc",
            "candidate_docs": candidate_docs,
            "parents": parents,
            "children": [],
            "graph_facts": concept_lines,
            "summaries": summaries,
            "shared_concepts": shared,
        }

    def _question_entity_candidates(self, question: str) -> List[str]:
        """Cheap entity hints: registry entity names found in the question."""
        q = (question or "").lower()
        found = []
        for meta in self.documents.values():
            for ent in meta.get("entities") or []:
                name = str(ent)
                if len(name) >= 3 and name.lower() in q and name not in found:
                    found.append(name)
        return found[:15]
    # Cheap entity hints: registry entity names found in the question.

    def select_candidate_docs(
        self, question: str, force_all: bool = False
    ) -> List[str]:
        """
        Shrink search space via Neo4j entity matches, then registry keyword fallback.
        Returns list of doc_ids (may be empty → caller treats as global).
        """
        if force_all or not self.documents:
            return list(self.documents.keys())

        q_lower = (question or "").lower()
        entity_names = self._question_entity_candidates(question)

        neo_docs: Set[str] = set()
        if entity_names:
            try:
                driver = self._get_driver()
                with driver.session() as session:
                    result = session.run(
                        """
                        MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
                        WHERE toLower(e.name) IN $names
                        RETURN DISTINCT c.doc_id AS doc_id
                        """,
                        names=[n.lower() for n in entity_names],
                    )
                    for row in result:
                        if row["doc_id"]:
                            neo_docs.add(row["doc_id"])
            except Exception as e:
                logger.warning("Neo4j candidate lookup failed: %s", e)

        if neo_docs:
            return [d for d in neo_docs if d in self.documents]

        # Registry metadata / keyword fallback
        scored: List[Tuple[int, str]] = []
        for doc_id, meta in self.documents.items():
            score = 0
            blob = " ".join(
                [
                    meta.get("doc_name", ""),
                    meta.get("summary", ""),
                    " ".join(meta.get("topics") or []),
                    " ".join(meta.get("keywords") or []),
                    " ".join(meta.get("entities") or []),
                ]
            ).lower()
            for token in re.findall(r"[a-z0-9]{3,}", q_lower):
                if token in blob:
                    score += 1
            if score:
                scored.append((score, doc_id))
        scored.sort(reverse=True)
        if scored:
            return [d for _, d in scored[:SYNTHESIS_TOP_M]]

        # Last resort: all docs (caller may still over-fetch globally for lookup)
        return []

    def _parent_record(self, parent_id: str) -> Optional[Dict[str, str]]:
        meta = self.parents.get(parent_id)
        if not meta:
            return None
        if isinstance(meta, dict):
            return {
                "id": parent_id,
                "text": meta.get("text", ""),
                "doc_id": meta.get("doc_id", ""),
                "doc_name": meta.get("doc_name", ""),
            }
        return {"id": parent_id, "text": meta, "doc_id": "", "doc_name": ""}

    def _faiss_search( # this is a function to search the vectorstore for the most relevant documents based on the question
        self, question: str, k: int, doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[Any, float]]:
        if self.vectorstore is None:
            return []
        if doc_ids:
            # Filter in Python — FAISS metadata filter support varies by version
            hits = self.vectorstore.similarity_search_with_score(question, k=max(k * 4, 40))
            allowed = set(doc_ids)
            filtered = [
                (d, s) for d, s in hits if d.metadata.get("doc_id") in allowed
            ]
            return filtered[:k]
        return self.vectorstore.similarity_search_with_score(question, k=k)

    def _rerank_children( # this is a function to rerank the children based on the question and the hits
        self, question: str, hits: List[Tuple[Any, float]]
    ) -> List[Tuple[Any, float]]:
        if not hits:
            return []
        try:
            pairs = [[question, doc.page_content] for doc, _ in hits]
            scores = self._get_reranker().predict(pairs)
            ranked = sorted(
                zip(hits, scores), key=lambda x: float(x[1]), reverse=True
            )
            return [(doc, float(score)) for (doc, _), score in ranked]
        except Exception as e:
            logger.warning("Rerank failed, using vector order: %s", e)
            return hits

    def lookup_retrieve(self, question: str) -> Dict[str, Any]:
        if self.vectorstore is None:
            raise ValueError("No vectorstore available. Please process documents first.")

        candidate_docs = self.select_candidate_docs(question, force_all=False)
        hits = self._faiss_search(question, OVERFETCH_K, candidate_docs or None)
        # Soft distance prune before rerank (FAISS L2); keep at least a few
        pruned = [(d, s) for d, s in hits if s <= MAX_L2_DISTANCE]
        hits = pruned if len(pruned) >= 3 else hits
        ranked = self._rerank_children(question, hits)[:TOP_K_CHILDREN]

        parent_ids: List[str] = []
        child_hits = []
        for doc, score in ranked:
            parent_id = doc.metadata.get("parent_id")
            child_hits.append(
                {
                    "child_id": doc.metadata.get("child_id"),
                    "parent_id": parent_id,
                    "doc_id": doc.metadata.get("doc_id"),
                    "doc_name": doc.metadata.get("doc_name"),
                    "score": float(score),
                    "text": doc.page_content,
                }
            )
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)
            if len(parent_ids) >= MAX_PARENTS:
                break

        parent_contexts = [
            p for pid in parent_ids if (p := self._parent_record(pid)) is not None
        ]
        parent_contexts = self._budget_parents(parent_contexts, CONTEXT_CHAR_BUDGET)

        graph_facts: List[str] = []
        try:
            graph_facts = self.get_graph_neighborhood(parent_ids, question)
        except Exception as e:
            logger.warning(f"Graph retrieval failed: {e}")

        return {
            "mode": "lookup",
            "candidate_docs": candidate_docs,
            "parents": parent_contexts,
            "children": child_hits,
            "graph_facts": graph_facts,
            "summaries": [],
        }

    def synthesis_retrieve(self, question: str) -> Dict[str, Any]:
        if self.vectorstore is None:
            raise ValueError("No vectorstore available. Please process documents first.")

        wants_every = bool(
            re.search(r"\b(each|every|all)\b", question or "", re.IGNORECASE)
        )
        candidate_docs = self.select_candidate_docs(
            question, force_all=wants_every and len(self.documents) <= SYNTHESIS_TOP_M
        )
        if not candidate_docs:
            candidate_docs = list(self.documents.keys())[:SYNTHESIS_TOP_M]
        else:
            candidate_docs = candidate_docs[:SYNTHESIS_TOP_M]

        summaries = []
        parents: List[Dict[str, str]] = []
        for doc_id in candidate_docs:
            meta = self.documents.get(doc_id) or {}
            summaries.append(
                {
                    "doc_id": doc_id,
                    "doc_name": meta.get("doc_name", doc_id),
                    "summary": meta.get("summary", ""),
                }
            )
            hits = self._faiss_search(question, k=3, doc_ids=[doc_id])
            if not hits:
                # fallback: first parent of this doc
                for pid, pmeta in self.parents.items():
                    if isinstance(pmeta, dict) and pmeta.get("doc_id") == doc_id:
                        rec = self._parent_record(pid)
                        if rec:
                            parents.append(rec)
                        break
                continue
            parent_id = hits[0][0].metadata.get("parent_id")
            rec = self._parent_record(parent_id) if parent_id else None
            if rec:
                parents.append(rec)

        parents = self._budget_parents(parents, CONTEXT_CHAR_BUDGET // 2)
        return {
            "mode": "synthesis",
            "candidate_docs": candidate_docs,
            "parents": parents,
            "children": [],
            "graph_facts": [],
            "summaries": summaries,
        }

    @staticmethod
    def _budget_parents(
        parents: List[Dict[str, str]], budget: int
    ) -> List[Dict[str, str]]:
        out, used = [], 0
        for p in parents:
            text = p.get("text", "")
            if used + len(text) > budget and out:
                break
            if len(text) > budget - used:
                clipped = dict(p)
                clipped["text"] = text[: max(0, budget - used)]
                out.append(clipped)
                break
            out.append(p)
            used += len(text)
        return out

    def hybrid_retrieve(self, question: str) -> Dict[str, Any]:
        mode = self.route_query(question)
        if mode == "cross_doc":
            return self.cross_doc_retrieve(question)
        if mode == "synthesis":
            return self.synthesis_retrieve(question)
        return self.lookup_retrieve(question)

    def generate_grounded_answer(
        self, question: str, retrieval: Dict[str, Any]
    ) -> str:
        parents = retrieval.get("parents") or []
        graph_facts = retrieval.get("graph_facts") or []
        summaries = retrieval.get("summaries") or []
        shared = retrieval.get("shared_concepts") or []
        mode = retrieval.get("mode", "lookup")

        if not parents and not summaries and not shared:
            return (
                "I don't know based on the provided documents — "
                "no sufficiently relevant passages were retrieved."
            )

        context_blocks = "\n\n".join(
            f"[{p.get('doc_name') or p['id']}|{p['id']}"
            f"{('|concept:' + p['concept']) if p.get('concept') else ''}]\n{p['text']}"
            for p in parents
        )
        summary_block = (
            "\n\n".join(
                f"[{s['doc_name']}]\n{s['summary']}" for s in summaries if s.get("summary")
            )
            or "(none)"
        )
        graph_block = (
            "\n".join(f"- {f}" for f in graph_facts) if graph_facts else "(no graph facts)"
        )
        shared_block = (
            "\n".join(
                f"- {c['name']} (in: {', '.join(c.get('doc_names') or [])})"
                for c in shared
            )
            or "(none found)"
        )
        history_block = (
            "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.chat_history[-4:])
            or "(none)"
        )

        structure = (
            "Format the answer as structured markdown: short headings and bullet points. "
            "No walls of paragraph text.\n"
            "Cite document names like [DocName] (and chunk ids if helpful).\n"
        )
        if mode == "cross_doc":
            grounding = (
                "Answer ONLY using Shared concepts, Document summaries, and Context excerpts.\n"
                "Use ONE markdown section per shared concept from the Shared concepts list.\n"
                "Under each concept, explain how the documents align or differ; cite [DocName].\n"
                "ONLY discuss concepts listed under Shared concepts — do not invent new ones.\n"
                "Do NOT invent mnemonics, exam templates, parameter counts, or tables "
                "that are not present in the provided materials.\n"
                "If an excerpt is too thin for a document, say "
                "\"not enough detail in retrieved excerpt for [DocName]\".\n"
            )
        elif mode == "synthesis":
            grounding = (
                "Answer ONLY using Document summaries and Context excerpts.\n"
                "Cover each selected document when the question asks about each/all.\n"
                "If a document lacks evidence for part of the question, say so under that document.\n"
                "Do not invent mnemonics, tables, or numeric details absent from the materials.\n"
            )
        else:
            grounding = (
                "Answer ONLY using the Context and Graph facts.\n"
                "If evidence is insufficient, say you don't know.\n"
            )

        prompt = (
            "You are a document QA assistant.\n"
            f"{grounding}"
            f"{structure}"
            "Do not invent facts not present in the provided materials.\n\n"
            f"Chat history:\n{history_block}\n\n"
            f"Shared concepts:\n{shared_block}\n\n"
            f"Document summaries:\n{summary_block}\n\n"
            f"Context:\n{context_blocks or '(none)'}\n\n"
            f"Graph facts:\n{graph_block}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        return self._llm_text(
            self._get_llm(temperature=0.1).invoke(prompt).content
        ).strip()

    def _regenerate_grounded(
        self, question: str, answer: str, retrieval: Dict[str, Any]
    ) -> str:
        parents = retrieval.get("parents") or []
        summaries = retrieval.get("summaries") or []
        shared = retrieval.get("shared_concepts") or []
        context = "\n\n".join(
            f"[{p.get('doc_name')}|{p['id']}]\n{p.get('text', '')}" for p in parents
        )[:8000]
        summary_text = "\n\n".join(
            f"{s.get('doc_name')}: {s.get('summary')}" for s in summaries
        )[:4000]
        shared_block = "\n".join(
            f"- {c['name']} ({', '.join(c.get('doc_names') or [])})" for c in shared
        )
        prompt = (
            "Rewrite the Draft answer so it is grounded ONLY in Shared concepts, "
            "Summaries, and Context. Drop every ungrounded claim. Keep structured markdown.\n"
            "Do not invent mnemonics, tables, or details absent from the materials.\n\n"
            f"Shared concepts:\n{shared_block or '(none)'}\n\n"
            f"Summaries:\n{summary_text or '(none)'}\n\n"
            f"Context:\n{context or '(none)'}\n\n"
            f"Question: {question}\n\n"
            f"Draft answer:\n{answer}\n\n"
            "Rewritten answer:"
        )
        return self._llm_text(
            self._get_llm(temperature=0).invoke(prompt).content
        ).strip()

    def check_faithfulness(
        self, answer: str, retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        parents = retrieval.get("parents") or []
        summaries = retrieval.get("summaries") or []
        graph_facts = retrieval.get("graph_facts") or []
        shared = retrieval.get("shared_concepts") or []
        mode = retrieval.get("mode", "lookup")

        if not parents and not summaries and not shared:
            return {"supported": False, "reason": "No context retrieved"}

        refuse_markers = ("i don't know", "i do not know", "insufficient")
        if any(m in answer.lower() for m in refuse_markers):
            return {"supported": True, "reason": "Explicit refusal"}

        context = "\n\n".join(p.get("text", "") for p in parents)
        summary_text = "\n\n".join(
            f"{s.get('doc_name')}: {s.get('summary')}" for s in summaries
        )
        facts = "\n".join(graph_facts)
        shared_text = "\n".join(
            f"{c.get('name')}: {', '.join(c.get('doc_names') or [])}" for c in shared
        )

        if mode in ("synthesis", "cross_doc"):
            # ponytail: summaries/shared lists are model-derived — weaker grounding ceiling
            prompt = (
                "Score whether the Answer is supported by Shared concepts, "
                "Document summaries, and Context excerpts.\n"
                "Claims grounded in those materials count as supported. "
                "Invented mnemonics/tables/counts not in materials => unsupported.\n"
                'Return JSON only: {"supported": true|false, "reason": "..."}\n\n'
                f"Shared concepts:\n{shared_text[:2000]}\n\n"
                f"Summaries:\n{summary_text[:4000]}\n\n"
                f"Context:\n{context[:4000]}\n\n"
                f"Answer:\n{answer}\n"
            )
        else:
            prompt = (
                "Score whether the Answer is fully supported by Context and Graph facts.\n"
                'Return JSON only: {"supported": true|false, "reason": "..."}\n'
                "supported=false if any material claim is not grounded.\n\n"
                f"Context:\n{context[:6000]}\n\n"
                f"Graph facts:\n{facts}\n\n"
                f"Answer:\n{answer}\n"
            )

        raw = self._llm_text(self._get_llm(temperature=0).invoke(prompt).content)
        data = self._parse_json_object(raw)
        if not data:
            return {
                "supported": True,
                "reason": "Could not parse faithfulness JSON; allowing answer",
            }
        return {
            "supported": bool(data.get("supported", False)),
            "reason": data.get("reason", ""),
        }

    def _ingest_one_document(
        self, raw_text: str, doc_name: str
    ) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        parents, children = self.get_parent_child_chunks(raw_text, doc_id, doc_name)
        parent_texts = [m["text"] for m in parents.values()]
        summary, topics, keywords = self.summarize_document(doc_name, parent_texts)
        entities = self.build_knowledge_graph(doc_id, doc_name, parents, summary=summary)
        registry = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "summary": summary,
            "topics": topics,
            "entities": entities,
            "keywords": keywords,
            "n_parents": len(parents),
            "n_children": len(children),
        }
        return parents, children, registry

    def process_documents(self, pdf_docs) -> dict:
        try:
            all_parents: Dict[str, Dict[str, str]] = {}
            all_children: List[Dict[str, str]] = []
            documents: Dict[str, Dict[str, Any]] = {}
            total_len = 0

            self.clear_knowledge_graph()

            for pdf in pdf_docs:
                name = getattr(pdf, "name", "uploaded.pdf")
                text = self._extract_pdf_text(pdf)
                if not text.strip():
                    logger.warning("No text extracted from %s", name)
                    continue
                total_len += len(text)
                parents, children, registry = self._ingest_one_document(text, name)
                all_parents.update(parents)
                all_children.extend(children)
                documents[registry["doc_id"]] = registry

            if not all_children:
                raise ValueError("No text could be extracted from the PDFs")

            self.parents = all_parents
            self.documents = documents
            self.create_vectorstore(all_children)
            self.save_index()
            self.chat_history = []
            self.ready = True

            logger.info(
                "Processed %d docs: %d parents, %d children",
                len(documents),
                len(all_parents),
                len(all_children),
            )
            return {
                "status": "success",
                "files_processed": len(documents),
                "chunks_created": len(all_children),
                "parents_created": len(all_parents),
                "text_length": total_len,
            }
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise

    def process_documents_from_paths(self, pdf_paths: List[str]) -> dict:
        try:
            all_parents: Dict[str, Dict[str, str]] = {}
            all_children: List[Dict[str, str]] = []
            documents: Dict[str, Dict[str, Any]] = {}
            total_len = 0

            self.clear_knowledge_graph()

            for path in pdf_paths:
                name = os.path.basename(path)
                with open(path, "rb") as file:
                    text = self._extract_pdf_text(file)
                if not text.strip():
                    continue
                total_len += len(text)
                parents, children, registry = self._ingest_one_document(text, name)
                all_parents.update(parents)
                all_children.extend(children)
                documents[registry["doc_id"]] = registry

            if not all_children:
                raise ValueError("No text could be extracted from the PDFs")

            self.parents = all_parents
            self.documents = documents
            self.create_vectorstore(all_children)
            self.save_index()
            self.chat_history = []
            self.ready = True
            return {
                "status": "success",
                "files_processed": len(documents),
                "chunks_created": len(all_children),
                "parents_created": len(all_parents),
                "text_length": total_len,
            }
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise

    def initialize_from_saved_vectorstore(self) -> bool:
        return self.load_index()

    def ask_question(self, question: str, conversation_id: Optional[str] = None) -> dict:
        try:
            if not question.strip():
                raise ValueError("Question cannot be empty")

            if not self.ready and not self.initialize_from_saved_vectorstore():
                raise ValueError("No index available. Please process documents first.")

            mode = self.route_query(question)
            logger.info("Processing question (%s): '%s'", mode, question)
            if mode == "cross_doc":
                retrieval = self.cross_doc_retrieve(question)
            elif mode == "synthesis":
                retrieval = self.synthesis_retrieve(question)
            else:
                retrieval = self.lookup_retrieve(question)

            answer = self.generate_grounded_answer(question, retrieval)
            faith = self.check_faithfulness(answer, retrieval)

            if not faith["supported"]:
                if mode in ("synthesis", "cross_doc"):
                    # Soft fail: one constrained regenerate, never hard I-don't-know dump
                    answer = self._regenerate_grounded(question, answer, retrieval)
                    faith = self.check_faithfulness(answer, retrieval)
                    if not faith["supported"]:
                        faith = {
                            "supported": False,
                            "reason": faith.get("reason", "")
                            or "Weakly grounded after regenerate",
                            "caution": True,
                        }
                else:
                    answer = (
                        "I don't know based on the provided documents. "
                        f"(Faithfulness check failed: {faith.get('reason', 'unsupported claims')})"
                    )

            self.chat_history.append((question, answer))

            return {
                "answer": answer,
                "status": "success",
                "conversation_id": conversation_id,
                "question": question,
                "route": retrieval.get("mode", mode),
                "candidate_docs": retrieval.get("candidate_docs") or [],
                "sources": retrieval.get("parents") or [],
                "summaries": retrieval.get("summaries") or [],
                "graph_facts": retrieval.get("graph_facts") or [],
                "shared_concepts": retrieval.get("shared_concepts") or [],
                "faithfulness": faith,
            }
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise

    def get_status(self) -> dict:
        return {
            "vectorstore_loaded": self.vectorstore is not None,
            "conversation_chain_initialized": self.ready or self.vectorstore is not None,
            "vectorstore_file_exists": os.path.exists(INDEX_FILE),
            "anthropic_api_configured": bool(self.anthropic_api_key),
            "neo4j_configured": bool(self.neo4j_uri and self.neo4j_password),
            "documents_indexed": len(self.documents),
        }


chatbot = PDFChatbot()


if __name__ == "__main__":
    sample = ("Section heading\n" + ("word " * 80) + "\n") * 20
    bot = PDFChatbot.__new__(PDFChatbot)
    bot.documents = {}
    parents, children = PDFChatbot.get_parent_child_chunks(
        bot, sample, "doc_test", "test.pdf"
    )
    assert parents and children
    assert all(c["parent_id"] in parents for c in children)
    assert all(c["doc_id"] == "doc_test" for c in children)
    assert all(p["doc_name"] == "test.pdf" for p in parents.values())
    assert PDFChatbot.route_query(bot, "What is two pointers?") == "lookup"
    assert (
        PDFChatbot.route_query(
            bot, "give me top questions from each and every pdf attached"
        )
        == "synthesis"
    )
    assert (
        PDFChatbot.route_query(
            bot,
            "identify every concept that appears in more than one document and how they differ",
        )
        == "cross_doc"
    )
    bot.documents = {"d1": {}, "d2": {}}
    assert (
        PDFChatbot.route_query(
            bot, "using all three uploaded documents, compare the concepts"
        )
        == "cross_doc"
    )
    # registry intersection fallback
    bot.documents = {
        "d1": {
            "doc_name": "a.pdf",
            "entities": ["ADL", "STRIPS"],
            "topics": [],
            "keywords": [],
        },
        "d2": {
            "doc_name": "b.pdf",
            "entities": ["ADL", "HTN"],
            "topics": [],
            "keywords": [],
        },
    }
    shared = PDFChatbot._registry_cross_doc_entities(bot, min_docs=2, limit=10)
    assert any(c["name"].lower() == "adl" for c in shared)
    assert not any(c["name"].lower() == "strips" for c in shared)
    sampled = PDFChatbot._sample_representative_parents(
        ["a" * 100, "b" * 100, "c" * 100, "d" * 100], 250
    )
    assert sampled and len(sampled) <= 250
    print(f"ok: {len(parents)} parents, {len(children)} children, cross_doc router ok")
