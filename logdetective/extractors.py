import os
import logging
import re

import drain3
from drain3.template_miner_config import TemplateMinerConfig
from llama_cpp import Llama, LlamaGrammar

from logdetective.constants import SUMMARIZE_PROMPT_TEMPLATE
from logdetective.utils import get_chunks, process_log

LOG = logging.getLogger("logdetective")

class RegexExtractor:
    """A class that extracts information from logs using dumb regexes
    https://stackoverflow.com/questions/6107700/tool-to-extract-java-stack-traces-from-log-files
    """
    def __init__(self):
        self.REGEX = re.compile("(^\tat |^Caused by: |^\t... \\d+ more)")
        # Usually, all inner lines of a stack trace will be "at" or "Caused by" lines.
        # With one exception: the line following a "nested exception is" line does not
        # follow that convention. Due to that, this line is handled separately.
        self.CONT = re.compile("; nested exception is: *$")

        self.exceptions = []

    def __call__(self, log: str) -> list[str]:
        self.process_log(log)
        return list(map(lambda exception: '\n'.join(exception[:5]), self.exceptions))

    def register_exception(self, exc: list[str]):
        self.exceptions.append(exc)

    def process_log(self, log: str):
        current_match = []
        last_line = None
        add_next_line = False
        for line in log.splitlines():
            if add_next_line and len(current_match) > 0:
                add_next_line = False
                current_match.append(line)
                continue
            match = self.REGEX.search(line) is not None
            if match and len(current_match) > 0:
                current_match.append(line)
            elif match:
                current_match.append(last_line)
                current_match.append(line)
            else:
                if len(current_match) > 0:
                    self.register_exception(current_match)
                current_match = []
            last_line = line
            add_next_line = self.CONT.search(line) is not None
        # If last line in file was a stack trace
        if len(current_match) > 0:
            self.register_exception(current_match)


class LLMExtractor:
    """
    A class that extracts relevant information from logs using a language model.
    """
    def __init__(self, model: Llama, n_lines: int = 2):
        self.model =  model
        self.n_lines = n_lines
        self.grammar = LlamaGrammar.from_string(
            "root ::= (\"Yes\" | \"No\")", verbose=False)

    def __call__(self, log: str, n_lines: int = 2, neighbors: bool = False) -> list[str]:
        chunks = self.rate_chunks(log)
        out = self.create_extract(chunks, neighbors)
        return out

    def rate_chunks(self, log: str) -> list[tuple]:
        """Scan log by the model and store results.

        :param log: log file content
        """
        results = []
        log_lines = log.split("\n")

        for i in range(0, len(log_lines), self.n_lines):
            block = '\n'.join(log_lines[i:i + self.n_lines])
            prompt = SUMMARIZE_PROMPT_TEMPLATE.format(log)
            out = self.model(prompt, max_tokens=7, grammar=self.grammar)
            out = f"{out['choices'][0]['text']}\n"
            results.append((block, out))

        return results

    def create_extract(self, chunks: list[tuple], neighbors: bool = False) -> list[str]:
        """Extract interesting chunks from the model processing.
        """
        interesting = []
        summary = []
        # pylint: disable=consider-using-enumerate
        for i in range(len(chunks)):
            if chunks[i][1].startswith("Yes"):
                interesting.append(i)
                if neighbors:
                    interesting.extend([max(i - 1, 0), min(i + 1, len(chunks) - 1)])

        interesting = set(interesting)

        for i in interesting:
            summary.append(chunks[i][0])

        return summary


class DrainExtractor:
    """A class that extracts information from logs using a template miner algorithm.
    """
    def __init__(self, verbose: bool = False, context: bool = False, max_clusters=8):
        config = TemplateMinerConfig()
        config.load(f"{os.path.dirname(__file__)}/drain3.ini")
        config.profiling_enabled = verbose
        config.drain_max_clusters = max_clusters
        self.miner = drain3.TemplateMiner(config=config)
        self.verbose = verbose
        self.context = context

    def __call__(self, log: str) -> list[str]:
        out = []
        for chunk in get_chunks(log):
            processed_line = self.miner.add_log_message(chunk)
            LOG.debug(processed_line)
        sorted_clusters = sorted(self.miner.drain.clusters, key=lambda it: it.size, reverse=True)
        for chunk in get_chunks(log):
            cluster = self.miner.match(chunk, "always")
            if cluster in sorted_clusters:
                out.append(chunk)
                sorted_clusters.remove(cluster)
        return out
