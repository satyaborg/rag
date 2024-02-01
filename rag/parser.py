import re
from pydantic import BaseModel
from typing import List, Union, Any
from enum import Enum

from llama_index import SimpleDirectoryReader
from llama_index.schema import Document
from llama_index.schema import IndexNode
from llama_index.node_parser import SentenceWindowNodeParser, SimpleNodeParser

from rag.common.types import ChunkingStrategy
from rag.common.exceptions import DocParserError
from rag.config import ParseConfig


class DocParser:

    def __init__(self, config: ParseConfig) -> None:
        self.config = config

    def assign_node_id(self, nodes: Union[List[IndexNode], List[dict]]) -> List[dict]:
        """Assigns custom (deterministic) node id to nodes."""
        for i, node in enumerate(nodes):
            custom_node_id = f"node_{i}"
            node.id_ = custom_node_id
        return nodes

    def load_docs(self, file_path: str = "./data") -> List[Document]:
        """Loads documents from a file path."""
        try:
            docs = SimpleDirectoryReader(
                input_dir=file_path, filename_as_id=True
            ).load_data()

            # sanitize docs
            for doc in docs:
                doc.text = self._sanitize(doc)

            # filter chunks based on stop words
            docs = [
                doc
                for doc in docs
                if doc.text.strip().lower() not in self.config.stop_words
            ]
            return docs
        except Exception as e:
            raise DocParserError(f"Error loading file: {file_path}") from e

    def _get_base_nodes(self, docs: List[Document]) -> List[dict]:
        """Gets base nodes from simple node parser."""
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.config.chunk_size,
            include_metadata=True,
        )
        base_nodes = node_parser.get_nodes_from_documents(docs)
        base_nodes = self.assign_node_id(base_nodes)
        return base_nodes

    def _get_window_nodes(self, docs: List[Document]) -> None:
        """Initialize window node parser."""
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=self.config.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        sentence_nodes = node_parser.get_nodes_from_documents(docs)
        return sentence_nodes

    def _get_child_to_parent_nodes(self, base_nodes: List[dict]) -> List[IndexNode]:
        """Gets child to parent nodes for small-to-large chunking."""
        all_nodes = []
        sub_node_parsers = [  # NOTE: include SimpleWindowNodeParser
            SimpleNodeParser.from_defaults(chunk_size=chunk_size)
            for chunk_size in self.config.chunk_sizes
        ]
        for base_node in base_nodes:
            for n in sub_node_parsers:
                sub_nodes = n.get_nodes_from_documents([base_node])
                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                all_nodes.extend(sub_inodes)

            # also add back the original node
            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(original_node)

        all_nodes_dict = {n.node_id: n for n in all_nodes}
        return all_nodes_dict

    def _sanitize(self, doc: Document) -> str:
        """Sanitizes and removes footer and leading numerals."""
        content = doc.get_content()
        content = re.sub(self.config.regex_footer, "", content)
        content = re.sub(
            self.config.regex_leading_numerals,
            lambda m: re.sub(r"^\d+", "", m.group(0)),
            content,
        )
        return content.strip()

    def get_nodes(
        self, docs: List[Document], chunking_strategy: str = ChunkingStrategy.BASE
    ) -> List[dict]:
        """Parses, chunks and returns nodes for documents, given a chunking strategy."""
        try:
            if chunking_strategy == ChunkingStrategy.BASE:
                return self._get_base_nodes(docs)
            elif chunking_strategy == ChunkingStrategy.WINDOW:
                return self._get_window_nodes(docs)
            elif chunking_strategy == ChunkingStrategy.CHILD_TO_PARENT:
                base_nodes = self._get_base_nodes(docs)
                return self._get_child_to_parent_nodes(base_nodes)
            else:
                raise ValueError(f"Chunking strategy {chunking_strategy} not found.")

        except Exception as e:
            print(e)
            raise DocParserError(f"Error parsing data") from e
