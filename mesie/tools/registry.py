"""Native MESIE / MAESI / NeuroAIX tool and skill registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NativeTool:
    id: str
    name: str
    skill_name: str
    category: str
    description: str
    triggers: List[str]
    command: str
    deliverable: Optional[str] = None
    requires: List[str] = field(default_factory=list)


TOOLS: List[NativeTool] = [
    # --- Core MESIE ---
    NativeTool("validate", "Validate Record", "mesie-validate", "core",
               "Validate spectral JSON against MESIE schema levels 1-6.",
               ["validate", "schema", "quality check"], "python examples/01_load_and_validate.py"),
    NativeTool("match", "Match Two Spectra", "mesie-match", "core",
               "Compare two spectral records with composite scoring.",
               ["match", "similarity", "compare spectra"], "python examples/02_match_two_records.py"),
    NativeTool("generate-psd", "Generate PSD", "mesie-generate", "core",
               "Synthesize power spectral density record (seeded).",
               ["generate psd", "synthetic psd"], "python examples/03_generate_psd.py"),
    NativeTool("generate-fas", "Generate FAS", "mesie-generate", "core",
               "Synthesize Fourier amplitude spectrum.",
               ["generate fas"], "python examples/04_generate_fas.py"),
    NativeTool("rotdnn", "RotDNN Workflow", "mesie-generate", "core",
               "RotDNN orientation spectral workflow.",
               ["rotdnn", "orientation"], "python examples/05_rotdnn_workflow.py"),
    NativeTool("embed", "Create Embedding", "mesie-embed", "core",
               "Vectorize a spectral record with SpectralVectorizer.",
               ["embed", "vectorize", "embedding"], "python examples/06_create_spectral_embedding.py"),
    NativeTool("bundled-data", "Load Bundled Data", "mesie-data", "core",
               "Load references, benchmarks, spectral library.",
               ["bundled data", "references", "benchmarks"], "python examples/09_load_bundled_data.py"),
    NativeTool("rank", "Rank Candidates", "mesie-match", "core",
               "Rank a query spectrum against a candidate pool (composite scoring).",
               ["rank", "rank candidates", "retrieval", "top-k"],
               "python scripts/run_rank_demo.py"),
    NativeTool("fix-data", "Fix Reference Data", "mesie-data", "core",
               "Repair bundled reference JSON (negative amplitudes, schema drift).",
               ["fix data", "repair references", "data quality"],
               "python scripts/fix_reference_data.py"),
    # --- Fingerprint / ANN ---
    NativeTool("fingerprint", "Fingerprint ANN Pipeline", "mesie-fingerprint", "retrieval",
               "TF → salient → LSH → approximate nearest neighbors.",
               ["fingerprint", "lsh", "ann", "time-frequency", "salient"],
               "python examples/13_fingerprint_ann_pipeline.py",
               "library/spectral_index.json"),
    NativeTool("embed-library", "Embed Full Library", "mesie-embed-library", "retrieval",
               "Embed all references + benchmarks into spectral_index.json.",
               ["embed library", "index corpus"], "python scripts/embed_spectral_library.py",
               "library/spectral_index.json"),
    NativeTool("embed-mine", "Embed User Library", "mesie-embed-library", "retrieval",
               "Embed your JSON folder into my_spectral_index.json.",
               ["my library", "embed folder"], "python scripts/embed_my_library.py <folder> --octopus",
               "library/my_spectral_index.json"),
    # --- Octopus / Internal API ---
    NativeTool("octopus", "Octopus Engineering", "mesie-octopus", "orchestration",
               "Eight-arm controller; EMBED/MATCH default to AISVectorPolyglot (vector + Rust/Python).",
               ["octopus", "eight arms", "internal api", "multi-arm", "polyglot arms"],
               "python examples/11_octopus_internal_api.py"),
    NativeTool("ais-polyglot", "AISVectorPolyglot Suite", "mesie-polyglot", "orchestration",
               "Polyglot test/use/integration: Python, Rust, Julia, Motoko, TypeScript + vector + AIS + 3rd-party AI.",
               ["ais", "polyglot", "julia", "rust", "motoko", "typescript", "vector bridge", "third party ai"],
               "python scripts/run_ais_polyglot_suite.py",
               "deliverables/AISVectorPolyglot_Integration_Report.json"),
    NativeTool("internal-bus", "Internal API Bus", "mesie-internal", "orchestration",
               "Register all engines on InternalBus; validate + match routing demo.",
               ["internal bus", "internal api", "engine bus", "message envelope"],
               "python scripts/run_internal_bus_demo.py"),
    NativeTool("engines", "List Engines", "mesie-internal", "orchestration",
               "Print registered engine names from default registry.",
               ["engines", "engine registry", "nine engines"],
               'python -c "from mesie.engines.registry import build_default_registry; r=build_default_registry(); print(\', \'.join(sorted(r.names())))"'),
    # --- MAESI SDK ---
    NativeTool("logic-prover", "SOLUS Logic Prover", "mesie-logic-prover", "solus",
               "Your local math proof caretaker — mini heart + mini brain, zero 3rd party.",
               ["logic prover", "proof", "theorem", "solus math"], "python scripts/run_solus_math_caretakers.py",
               "deliverables/SOLUS_Math_Caretakers_Report.json"),
    NativeTool("pattern-forge", "SOLUS Pattern Forge", "mesie-pattern-forge", "solus",
               "Your X-ray math caretaker — z-depth, spectral decompose, phi-harmonics, fully local.",
               ["pattern forge", "xray", "z-depth", "spectral decompose", "solus math"],
               "python scripts/run_solus_math_caretakers.py",
               "deliverables/SOLUS_Math_Caretakers_Report.json"),
    NativeTool("solus-organism", "SOLUS SDK Organism", "mesie-solus-organism", "solus",
               "Hosts Logic Prover + Pattern Forge as autonomous caretakers inside MAESI SDK.",
               ["solus", "organism", "caretaker", "mini heart", "mini brain", "sovereign"],
               "python scripts/run_solus_math_caretakers.py",
               "deliverables/SOLUS_Math_Caretakers_Report.json"),
    NativeTool("maesi", "MAESI SDK Run", "mesie-maesi", "maesi",
               "Run MAESI v1.1: laws, elements, bio, technical + research knowledge, fast compute.",
               ["maesi", "maesi sdk", "neuroaix sdk"], "python scripts/run_maesi_sdk.py",
               "deliverables/MAESI_SDK_Run_Report.json"),
    NativeTool("knowledge", "Research Knowledge Search", "mesie-knowledge", "maesi",
               "Search technical + research catalogs.",
               ["research", "technical library", "knowledge"],
               'python -c "from mesie.sdk import search_research; print(search_research(\'LSH ANN\', 5))"'),
    NativeTool("fast-compute", "Fast Compute Benchmark", "mesie-maesi", "maesi",
               "Batch matrix cosine vs loop match — virtual-chip throughput.",
               ["fast compute", "batch match", "speedup", "virtual chip"],
               "python scripts/run_fast_compute_benchmark.py",
               "deliverables/MAESI_Fast_Compute_Benchmark.json"),
    # --- NeuroAIX ---
    NativeTool("neuroaix", "NeuroAIX Connectome", "mesie-neuroaix", "neuroaix",
               "MAESI observation encoder + 3D connectome brain demo.",
               ["neuroaix", "connectome", "brain regions"],
               "python examples/08_3d_connectome_brain.py"),
    NativeTool("cognitive", "Cognitive Memory", "mesie-neuroaix", "neuroaix",
               "Spectral memory adapter and agent state.",
               ["cognitive", "memory adapter"], "python examples/07_cognitive_memory_adapter.py"),
    # --- Domain analysis ---
    NativeTool("domains", "Multi-Domain Suites", "mesie-domains", "analysis",
               "Terrain, robotics, orbital, power, seismic analysis suites.",
               ["domains", "terrain", "robotics", "power", "seismic suites"],
               "python scripts/run_multi_domain_suites.py",
               "deliverables/MESIE_Multi_Domain_Suite_Report.md"),
    NativeTool("orbital", "Orbital Edge 50d", "mesie-orbital", "analysis",
               "50 days back + 50 forward orbital-edge analysis.",
               ["orbital", "satellite edge", "50 day"], "python scripts/orbital_edge_50d_analysis.py",
               "scripts/orbital_edge_50d_report.json"),
    NativeTool("monte-carlo", "Monte Carlo Enterprise", "mesie-enterprise", "analysis",
               "Monte Carlo benchmark across 10 enterprise use cases.",
               ["monte carlo", "enterprise", "sla benchmark"],
               "python scripts/monte_carlo_enterprise_benchmark.py --trials 200",
               "deliverables/MESIE_Monte_Carlo_Enterprise_Report.md"),
    NativeTool("laptop", "Laptop Research Report", "mesie-laptop", "analysis",
               "Virtual chip framing + embedded library report.",
               ["laptop", "virtual chip", "research report"],
               "python scripts/embed_spectral_library.py && python scripts/generate_laptop_research_report.py",
               "deliverables/MESIE_Laptop_Research_Report.md"),
    # --- QA / Speed ---
    NativeTool("test", "Run Test Suite", "mesie-test", "qa",
               "Run full pytest suite.",
               ["test", "pytest", "ci"], "python -m pytest tests/ -q"),
    NativeTool("benchmark", "Speed Benchmark", "mesie-benchmark", "qa",
               "Determinism and match/embed throughput.",
               ["benchmark", "speed", "determinism"], "python scripts/determinism_benchmark.py"),
    NativeTool("sdk-drive", "SDK Test Drive", "mesie-test", "qa",
               "End-to-end SDK smoke with report JSON.",
               ["sdk test drive", "smoke test"], "python scripts/sdk_test_drive.py",
               "scripts/sdk_test_drive_report.json"),
    # --- Deploy ---
    NativeTool("cloudflare", "Cloudflare Worker", "mesie-deploy", "deploy",
               "Edge API deploy guide and worker health.",
               ["cloudflare", "worker", "edge api", "deploy"],
               "type docs\\cloudflare.md"),
    NativeTool("catalog", "Export Tool Catalog", "mesie-deploy", "deploy",
               "Write JSON catalog of all native tools and skills.",
               ["catalog", "tool list", "skills map"],
               "python -m mesie.tools.cli catalog",
               "deliverables/MESIE_Native_Tools_Catalog.json"),
    # --- Research Agent & Thesis Engine ---
    NativeTool("research-agent", "Research Agent", "mesie-research", "research",
               "Autonomous multi-step research: plan, execute, report.",
               ["research", "research agent", "investigate", "literature"],
               'python -c "from mesie.research import ResearchAgent; a=ResearchAgent(); print(a.research(\'spectral analysis methods\').to_dict())"'),
    NativeTool("thesis", "Thesis Engine", "mesie-research", "research",
               "Hypothesis → experiment → analysis → conclusion pipeline.",
               ["thesis", "hypothesis", "experiment", "scientific method"],
               'python -c "from mesie.research import ThesisEngine; e=ThesisEngine(); print(e.run(\'Test\', \'H0\').to_dict())"'),
    NativeTool("research-plan", "Research Planner", "mesie-research", "research",
               "Decompose a question into executable task graph.",
               ["plan research", "decompose", "task graph"],
               'python -c "from mesie.research import ResearchPlanner; p=ResearchPlanner(); print(p.plan(\'test\').summary())"'),
    # --- Labs ---
    NativeTool("lab-spectral", "Spectral Lab", "mesie-lab", "labs",
               "Spectral analysis lab: validate, match, generate, embed.",
               ["spectral lab", "spectrum lab"],
               'python -c "from mesie.labs import SpectralLab; print(SpectralLab().capabilities)"'),
    NativeTool("lab-chemistry", "Chemistry Lab", "mesie-lab", "labs",
               "Molecular fingerprints, similarity, reactions, compound lookup.",
               ["chemistry lab", "molecular", "compound"],
               'python -c "from mesie.labs import ChemistryLab; print(ChemistryLab().run(\'formula_parse\', formula=\'H2O\').to_dict())"'),
    NativeTool("lab-physics", "Physics Lab", "mesie-lab", "labs",
               "Constants, unit conversion, kinematics, thermodynamics, waves.",
               ["physics lab", "constants", "unit convert"],
               'python -c "from mesie.labs import PhysicsLab; print(PhysicsLab().run(\'constants\', name=\'c\').to_dict())"'),
    NativeTool("lab-bio", "Bio Lab", "mesie-lab", "labs",
               "Bioinformatics: GC content, translation, alignment, k-mers.",
               ["bio lab", "bioinformatics", "dna", "protein"],
               'python -c "from mesie.labs import BioLab; print(BioLab().run(\'gc_content\', sequence=\'ATCGGCTA\').to_dict())"'),
    NativeTool("lab-earth", "Earth Science Lab", "mesie-lab", "labs",
               "Seismic, geospatial, atmosphere, soil classification.",
               ["earth lab", "seismic", "geospatial", "earthquake"],
               'python -c "from mesie.labs import EarthLab; print(EarthLab().run(\'seismic_magnitude\', magnitude=5.0).to_dict())"'),
    # --- Hub ---
    NativeTool("hub", "Research Hub", "mesie-hub", "hub",
               "Universal research platform: connects all labs, agents, and tools.",
               ["hub", "research hub", "universal lab", "platform"],
               'python -c "from mesie.hub import ResearchHub; h=ResearchHub(); print(h.info())"'),
    # --- Data Sources ---
    NativeTool("data-arxiv", "ArXiv Source", "mesie-data-sources", "data_sources",
               "Search and fetch scientific papers from ArXiv.",
               ["arxiv", "papers", "preprints"],
               'python -c "from mesie.data_sources import ArxivSource; print(ArxivSource().search(\'spectral\'))"'),
    NativeTool("data-pubchem", "PubChem Source", "mesie-data-sources", "data_sources",
               "Chemical compound lookup from PubChem (NIH).",
               ["pubchem", "compound", "chemical database"],
               'python -c "from mesie.data_sources import PubChemSource; print(PubChemSource().search_compound(\'aspirin\'))"'),
    NativeTool("data-usgs", "USGS Source", "mesie-data-sources", "data_sources",
               "Earthquake catalog, geology, water data from USGS.",
               ["usgs", "earthquake data", "geological"],
               'python -c "from mesie.data_sources import USGSSource; print(USGSSource().available_datasets())"'),
]


SKILL_CATEGORIES = {
    "solus": "SOLUS Local Math AI Caretakers",
    "core": "MESIE Core Spectral Engine",
    "retrieval": "Embedding, Fingerprint & ANN Retrieval",
    "orchestration": "Octopus & Internal API",
    "maesi": "MAESI SDK & Knowledge",
    "neuroaix": "NeuroAIX Connectome & Cognitive",
    "analysis": "Domain & Enterprise Analysis",
    "qa": "Quality Assurance & Benchmarks",
    "deploy": "Deploy & Edge",
    "research": "Research Agent & Thesis Engine",
    "labs": "Multi-Domain Research Labs",
    "hub": "Universal Research Hub",
    "data_sources": "External Data Sources",
}


def tool_by_id(tool_id: str) -> Optional[NativeTool]:
    for t in TOOLS:
        if t.id == tool_id:
            return t
    return None


def tools_by_category(category: str) -> List[NativeTool]:
    return [t for t in TOOLS if t.category == category]


def all_skill_names() -> List[str]:
    return sorted({t.skill_name for t in TOOLS})