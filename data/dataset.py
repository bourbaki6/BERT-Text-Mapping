
import csv
import html
import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

#---corpus taken from chat gpt - 5 generated content---
DEMO_CORPUS = {
    "Machine Learning": [
        "Deep learning models achieve state-of-the-art results on image classification tasks.",
        "Gradient descent optimizes neural network weights by minimizing the loss function.",
        "Overfitting occurs when a model learns noise rather than the underlying signal.",
        "Transfer learning reuses pretrained model weights for new downstream tasks.",
        "Convolutional neural networks are particularly effective for spatial data like images.",
        "Batch normalization stabilizes training by normalizing activations per mini-batch.",
        "Dropout is a regularization technique that randomly zeroes activations during training.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
        "The transformer architecture replaced recurrent networks for most NLP tasks.",
        "Hyperparameter tuning can be automated using Bayesian optimization or grid search.",
        "Support vector machines find the maximum-margin hyperplane between classes.",
        "Random forests aggregate predictions from many decision trees to reduce variance.",
        "XGBoost is a gradient-boosted tree implementation widely used in competitions.",
        "K-nearest neighbours classifies points by a majority vote of their k nearest points.",
        "Principal component analysis reduces dimensionality by projecting onto eigenvectors.",
        "Autoencoders learn compressed latent representations in an unsupervised manner.",
        "Generative adversarial networks pit a generator against a discriminator.",
        "Reinforcement learning agents learn policies by maximizing cumulative reward.",
        "The bias-variance tradeoff describes the tension between model flexibility and error.",
        "Cross-validation estimates model generalization by rotating training and test sets.",
        "Feature engineering transforms raw inputs into more predictive representations.",
        "Label smoothing softens hard targets to prevent overconfident predictions.",
        "Curriculum learning starts training on easy examples before moving to hard ones.",
        "Meta-learning aims to learn how to learn from small amounts of data.",
        "Neural architecture search automates the design of model topologies.",
        "Quantization reduces model size by representing weights in lower precision.",
        "Pruning removes redundant neurons or connections to compress models.",
        "Knowledge distillation trains a small student model to mimic a large teacher.",
        "Multi-task learning shares representations across related objectives.",
        "Contrastive learning pulls together similar samples and pushes apart dissimilar ones.",
    ],
    "Finance": [
        "Stock prices fluctuate based on supply and demand dynamics in equity markets.",
        "Diversification reduces portfolio risk by spreading investments across asset classes.",
        "The Federal Reserve adjusts interest rates to control inflation and employment.",
        "Bond yields move inversely to bond prices in fixed-income markets.",
        "Hedge funds use leverage and short selling to generate absolute returns.",
        "Options give holders the right but not the obligation to buy or sell an asset.",
        "Market capitalization is calculated by multiplying share price by shares outstanding.",
        "Earnings per share measures profitability on a per-share basis.",
        "The price-to-earnings ratio compares a company's share price to its annual earnings.",
        "Venture capital funds invest in early-stage startups in exchange for equity.",
        "Inflation erodes the purchasing power of money over time.",
        "A bull market is characterized by rising asset prices and investor optimism.",
        "Credit default swaps are derivatives that transfer credit risk between parties.",
        "Quantitative easing involves central banks purchasing government bonds.",
        "Algorithmic trading uses computer programs to execute trades at high speed.",
        "The efficient market hypothesis argues prices reflect all available information.",
        "Arbitrage exploits price discrepancies across markets to earn risk-free profit.",
        "Liquidity refers to how easily an asset can be converted to cash without loss.",
        "ESG investing considers environmental, social, and governance factors.",
        "Dollar-cost averaging invests fixed amounts at regular intervals regardless of price.",
        "Compound interest grows exponentially as returns are reinvested over time.",
        "A recession is typically defined as two consecutive quarters of negative GDP growth.",
        "Cryptocurrency markets are highly volatile and operate around the clock.",
        "The yield curve plots interest rates across different bond maturities.",
        "Private equity acquires companies to restructure and resell at a profit.",
        "Dividend yield measures annual dividends relative to the share price.",
        "Portfolio rebalancing maintains target asset allocations by buying and selling.",
        "Tax-loss harvesting realizes losses to offset capital gains.",
        "Inflation-linked bonds adjust principal and coupons based on price indices.",
        "Sovereign wealth funds invest national savings in global financial assets.",
    ],
    "Climate Change": [
        "Rising global temperatures are attributed to increased greenhouse gas emissions.",
        "Carbon dioxide from fossil fuel combustion is the primary driver of warming.",
        "Melting Arctic ice amplifies warming through the ice-albedo feedback loop.",
        "Sea level rise threatens coastal cities and low-lying island nations.",
        "Renewable energy sources like solar and wind are expanding rapidly worldwide.",
        "The Paris Agreement aims to limit warming to 1.5 degrees Celsius above pre-industrial levels.",
        "Carbon capture technology removes CO2 from the atmosphere or industrial emissions.",
        "Deforestation releases stored carbon and reduces the planet's carbon-absorbing capacity.",
        "Ocean acidification harms marine ecosystems by absorbing excess atmospheric CO2.",
        "Extreme weather events including droughts and floods are becoming more frequent.",
        "Permafrost thaw in Siberia releases methane, a potent greenhouse gas.",
        "Electric vehicles reduce transport emissions when charged with clean electricity.",
        "Green hydrogen produced by electrolysis can decarbonize heavy industry.",
        "Carbon pricing assigns a cost to emissions to incentivize cleaner alternatives.",
        "Coral bleaching events are intensifying as ocean temperatures rise.",
        "Sustainable agriculture practices can sequester carbon in soils.",
        "Climate refugees are being displaced by rising seas and extreme heat.",
        "The IPCC publishes scientific consensus reports on climate change impacts.",
        "Net-zero targets require balancing emissions released with those removed.",
        "Geoengineering proposals like solar radiation management carry uncertain risks.",
        "Biodiversity loss is accelerating as habitats shift under changing climate conditions.",
        "Building retrofits improve energy efficiency and cut heating and cooling emissions.",
        "Offshore wind farms generate large amounts of clean electricity at lower land cost.",
        "Circular economy principles reduce waste and lower overall resource consumption.",
        "Heat pumps are more efficient than gas boilers for heating homes and buildings.",
        "Urban heat islands make cities hotter than surrounding rural areas.",
        "Climate litigation is holding governments and corporations accountable for inaction.",
        "Just transition policies protect workers in fossil fuel industries during decarbonization.",
        "Nature-based solutions use ecosystems like wetlands and forests to capture carbon.",
        "Nuclear energy provides low-carbon baseload power but faces public acceptance challenges.",
    ],
    "Space Exploration": [
        "NASA's Artemis program aims to return humans to the lunar surface by the mid-2020s.",
        "SpaceX's Falcon 9 rocket pioneered reusable booster technology to cut launch costs.",
        "The James Webb Space Telescope observes the universe in infrared at Lagrange point L2.",
        "Mars missions face challenges including radiation exposure and the six-month transit.",
        "The International Space Station has been continuously inhabited since October 2000.",
        "Exoplanets are detected using the transit method by observing stellar brightness dips.",
        "Black holes warp spacetime so strongly that not even light can escape their event horizon.",
        "Neutron stars are incredibly dense remnants of supernova explosions.",
        "Gravitational waves were first directly detected by LIGO in 2015.",
        "The Voyager probes have traveled beyond the heliopause into interstellar space.",
        "Asteroid mining could provide rare metals and water ice for future space operations.",
        "Starship is designed to carry 100 tonnes to orbit and eventually reach Mars.",
        "Lunar ice deposits at the poles could be converted to rocket propellant.",
        "Solar sails use photon pressure from sunlight to propel spacecraft without fuel.",
        "The Drake equation estimates the number of communicating civilizations in the galaxy.",
        "Titan's methane lakes make it a compelling target for astrobiology research.",
        "Europa's subsurface ocean is considered a candidate for extraterrestrial life.",
        "The Hubble Space Telescope has been fundamental to measuring the expansion of the universe.",
        "Cubesats enable low-cost access to orbit for universities and small organizations.",
        "Space debris in low-Earth orbit poses collision risks to operational satellites.",
        "Ion propulsion systems are highly efficient for long-duration deep space missions.",
        "The cosmic microwave background is the thermal radiation left over from the Big Bang.",
        "Dark matter does not interact with light but influences galactic structure through gravity.",
        "Orbital mechanics govern how spacecraft navigate between planets using gravity assists.",
        "Satellite imagery from orbit supports agriculture, disaster response, and urban planning.",
        "Radiation belts around Earth trap charged particles that are hazardous to astronauts.",
        "The Perseverance rover collects rock samples for a future Mars Sample Return mission.",
        "Commercial crew vehicles from SpaceX and Boeing transport astronauts to the ISS.",
        "Space tourism is becoming a market as costs decline and safety improves.",
        "Atmospheric entry requires heat shields to dissipate kinetic energy as thermal energy.",
    ],
    "Medicine": [
        "mRNA vaccines teach the immune system to recognize viral proteins without infection.",
        "CRISPR-Cas9 allows precise edits to DNA sequences by cutting at targeted locations.",
        "Antibiotics kill or inhibit bacterial growth by targeting cell wall synthesis or replication.",
        "Immunotherapy harnesses the patient's immune system to attack cancer cells.",
        "Clinical trials evaluate drug safety and efficacy in phases from small to large groups.",
        "The blood-brain barrier restricts passage of molecules from blood to brain tissue.",
        "Stem cells can differentiate into specialized cell types for regenerative therapies.",
        "Personalized medicine tailors treatments to individual genetic profiles.",
        "Antibiotic resistance is a global health crisis driven by overuse and misuse of drugs.",
        "Gut microbiome composition is linked to metabolic health, immunity, and mental health.",
        "Telemedicine expanded rapidly during the pandemic enabling remote consultations.",
        "Organ-on-a-chip technology replicates human tissue function for drug testing.",
        "Epidemiology tracks disease patterns to identify risk factors and causes.",
        "Herd immunity occurs when enough of a population is immune to slow disease spread.",
        "Radiotherapy destroys cancer cells by focusing ionizing radiation on tumors.",
        "Statins lower LDL cholesterol and reduce the risk of cardiovascular events.",
        "Type 2 diabetes is strongly linked to obesity, sedentary lifestyle, and genetics.",
        "Magnetic resonance imaging uses magnetic fields and radio waves to visualize soft tissue.",
        "Palliative care focuses on improving quality of life for patients with serious illness.",
        "Gene therapy delivers corrected genetic material to cells to treat hereditary disorders.",
        "Monoclonal antibodies are engineered to bind specific proteins like cancer antigens.",
        "Robotic surgery enables minimally invasive procedures with high precision.",
        "Blood pressure monitoring is critical for detecting and managing hypertension.",
        "Alzheimer's disease is characterized by amyloid plaques and tau tangles in the brain.",
        "Vaccines have eradicated smallpox and dramatically reduced polio incidence globally.",
        "Electronic health records improve care coordination and reduce medical errors.",
        "Prosthetics are advancing with neural interfaces that allow thought-controlled movement.",
        "Sleep disorders such as sleep apnea are associated with cardiovascular risk.",
        "Mental health conditions are increasingly recognized alongside physical health.",
        "Precision oncology identifies targetable mutations to guide cancer treatment selection.",
    ],
    "Cybersecurity": [
        "Phishing attacks trick users into revealing credentials through deceptive emails.",
        "Zero-day vulnerabilities are exploited before software vendors have issued a patch.",
        "Ransomware encrypts victims' files and demands payment for the decryption key.",
        "Multi-factor authentication significantly reduces account takeover risk.",
        "End-to-end encryption ensures only sender and recipient can read messages.",
        "Penetration testing simulates attacks to identify security weaknesses proactively.",
        "A firewall filters network traffic based on predefined security rules.",
        "SQL injection exploits unsanitized inputs to execute malicious database queries.",
        "Social engineering manipulates people into bypassing security controls.",
        "Intrusion detection systems monitor network traffic for suspicious patterns.",
        "Supply chain attacks compromise software dependencies to reach downstream targets.",
        "The principle of least privilege limits access rights to the minimum necessary.",
        "Bug bounty programs reward researchers for responsibly disclosing vulnerabilities.",
        "Threat intelligence sharing between organizations improves collective defence.",
        "Encryption at rest protects stored data if physical media is compromised.",
        "DDOS attacks overwhelm servers with traffic to make services unavailable.",
        "Security information and event management centralizes log analysis.",
        "Identity and access management controls who can access which resources.",
        "Cryptographic hash functions produce a fixed-size digest from arbitrary input.",
        "Public key infrastructure manages digital certificates for authentication.",
        "Container security must address image vulnerabilities and runtime privilege escalation.",
        "DevSecOps integrates security practices throughout the software development lifecycle.",
        "Network segmentation limits lateral movement after an attacker gains initial access.",
        "Insider threats from employees with malicious intent or poor security hygiene.",
        "Patch management applies software updates promptly to close known vulnerabilities.",
        "Threat modelling identifies potential attack vectors during system design.",
        "Honeypots deceive attackers to gather intelligence on their techniques.",
        "The MITRE ATT&CK framework documents adversary tactics, techniques, and procedures.",
        "Data loss prevention tools monitor and control sensitive data transfers.",
        "Quantum computing threatens to break current asymmetric encryption algorithms.",
    ],
    "Sports": [
        "Football is the world's most popular sport with billions of fans across every continent.",
        "The Olympics bring together athletes from over 200 nations every four years.",
        "Cricket has a massive following in South Asia, the UK, and Australia.",
        "Tennis Grand Slams are played on grass, clay, and hard court surfaces.",
        "NBA players earn among the highest salaries of any professional athletes worldwide.",
        "Formula One racing combines aerodynamics, engineering, and driver skill.",
        "Marathon running requires months of structured training and careful nutrition.",
        "Swimming technique emphasizes streamlined body position and efficient stroke mechanics.",
        "Team cohesion and communication are as important as individual skill in basketball.",
        "eSports tournaments now attract audiences rivaling traditional sporting events.",
        "Rugby union and rugby league differ in player numbers, rules, and scoring systems.",
        "Cycling's Grand Tours cover thousands of kilometres of mountain and flat stages.",
        "Sports analytics use data to optimize player selection, tactics, and training loads.",
        "Injury prevention programs incorporate strength, flexibility, and movement training.",
        "Goalkeepers in football must combine reflexes, positioning, and distribution skills.",
        "The World Cup is watched by a global audience of several billion people.",
        "Performance-enhancing drugs remain a persistent problem across elite sports.",
        "Altitude training boosts red blood cell production for endurance athletes.",
        "Youth academies develop talent from an early age in football and other sports.",
        "Sports psychology helps athletes manage pressure and maintain focus under stress.",
        "Ice hockey demands skating ability, puck handling, and aggressive forechecking.",
        "Volleyball strategy involves serve reception, setting, and aggressive spiking.",
        "Boxing weight classes ensure bouts are contested between evenly matched fighters.",
        "Athletics field events include javelin, discus, shot put, and high jump.",
        "Baseball statistics like WAR attempt to quantify a player's total contribution.",
        "Stadium design balances sightlines, acoustics, and safety for spectators.",
        "Sponsorship and broadcast rights generate billions in revenue for top leagues.",
        "Paralympic athletes compete at exceptional levels despite physical impairments.",
        "Coaching tactics evolve rapidly as video analysis makes opponent preparation easier.",
        "Transfer fees in elite football have escalated to hundreds of millions of euros.",
    ],
    "Cooking": [
        "The Maillard reaction produces the brown crust and complex flavour on seared meat.",
        "Emulsification creates stable mixtures of oil and water as in mayonnaise or hollandaise.",
        "Knife skills are foundational — precise cuts ensure even cooking times across ingredients.",
        "Fermentation transforms raw ingredients by harnessing beneficial microorganisms.",
        "Seasoning with salt enhances existing flavours rather than making food taste salty.",
        "Stock made from roasted bones forms the base of many classical French sauces.",
        "Braising cooks tough cuts slowly in liquid to break down collagen into gelatin.",
        "Sous vide cooking holds food at precise temperature in a water bath for consistent results.",
        "Pasta dough requires the correct hydration ratio and resting time to develop gluten.",
        "Baking is a precise science where small measurement errors can ruin texture and rise.",
        "Mise en place — having everything prepared and in place — is fundamental to professional cooking.",
        "Acidity from citrus or vinegar balances richness and brightens flavours in a dish.",
        "Umami is the fifth basic taste, found in soy sauce, aged cheese, and mushrooms.",
        "Tempering chocolate aligns cocoa butter crystals for a glossy snap.",
        "Resting meat after cooking allows juices to redistribute throughout the muscle.",
        "Blanching vegetables briefly in boiling water preserves colour and texture.",
        "Reduction concentrates flavours by evaporating liquid from sauces and stocks.",
        "Caramelization occurs when sugars break down under dry heat above 160 degrees Celsius.",
        "Gluten forms when wheat flour is hydrated and worked, providing structure to bread.",
        "Spice blooming in fat releases fat-soluble aromatic compounds for more flavour.",
        "Cold smoking adds flavour without cooking, while hot smoking cooks and flavours simultaneously.",
        "Yeast produces carbon dioxide and alcohol during fermentation, causing bread to rise.",
        "A roux of equal parts butter and flour thickens sauces when liquid is gradually added.",
        "Marinating tenderizes proteins and adds flavour through acids and enzymes.",
        "Japanese dashi stock is made from dried kombu seaweed and bonito flakes.",
        "Cooking vegetables in their own juices concentrates flavour through a fond.",
        "Chilli heat comes from capsaicin, which binds to heat receptors on the tongue.",
        "Fresh pasta and dried pasta have different ideal cooking times and textures.",
        "Buttercream icings rely on the ratio of fat to sugar for stability and spreadability.",
        "Pressure cookers cut cooking times dramatically by raising the boiling point of water.",
    ],
    "Artificial Intelligence": [
        "Large language models are trained on vast text corpora using self-supervised objectives.",
        "Prompt engineering involves crafting inputs that elicit the desired model behaviour.",
        "Chain-of-thought prompting improves reasoning by encouraging step-by-step solutions.",
        "Retrieval-augmented generation grounds language model outputs in retrieved documents.",
        "AI safety research aims to align model behaviour with human values and intentions.",
        "Fine-tuning adapts a pretrained model to a specific domain or task with less data.",
        "Token prediction is the fundamental training signal for most modern language models.",
        "Hallucination refers to language models generating confident but factually wrong content.",
        "Multimodal models process both text and images to perform cross-modal reasoning.",
        "Constitutional AI trains models using feedback derived from a set of ethical principles.",
        "Instruction following is tested by benchmarks that evaluate model compliance and accuracy.",
        "Sparse mixture-of-experts models activate only a fraction of parameters per token.",
        "Context window size determines how much text a language model can process at once.",
        "AI agents combine language models with tools like search and code execution.",
        "Model evaluation requires careful benchmark construction to avoid contamination.",
        "Interpretability research tries to understand what computations happen inside neural networks.",
        "Embedding models map text to dense vectors capturing semantic similarity.",
        "Semantic search uses embeddings to retrieve documents by meaning rather than keywords.",
        "RLHF uses human preference ratings to align language models via reinforcement learning.",
        "Open-source language models allow broader access and customization than proprietary ones.",
        "Model compression enables deployment on edge devices with limited compute.",
        "Tokenization splits text into subword units that the model processes as discrete tokens.",
        "Positional encodings allow transformers to represent sequence order.",
        "Flash attention reduces memory usage by recomputing activations rather than storing them.",
        "Vision transformers apply the attention mechanism to image patches rather than text tokens.",
        "AI regulation is developing globally in response to rapid capability improvements.",
        "Agentic AI systems autonomously plan and execute multi-step tasks with tools.",
        "Red-teaming probes models for harmful or unsafe outputs systematically.",
        "Synthetic data generation augments training sets where real data is scarce.",
        "Multiagent systems involve multiple AI models collaborating or competing on tasks.",
    ],
    "History": [
        "The Roman Empire at its peak controlled territory from Britain to Mesopotamia.",
        "The printing press invented by Gutenberg accelerated the spread of knowledge in Europe.",
        "The Industrial Revolution transformed Britain from an agrarian to a manufacturing economy.",
        "World War One was triggered by the assassination of Archduke Franz Ferdinand in Sarajevo.",
        "The French Revolution overthrew the monarchy and introduced the Declaration of Rights of Man.",
        "The Silk Road connected East Asia, Central Asia, the Middle East, and Europe for centuries.",
        "Ancient Egypt's pyramids were built as tombs for pharaohs and their consorts.",
        "The Black Death killed an estimated third of Europe's population in the 14th century.",
        "The Renaissance revived interest in classical learning, art, and humanism in Italy.",
        "Columbus's 1492 voyage initiated sustained contact between Europe and the Americas.",
        "The Mongol Empire was the largest contiguous land empire in human history.",
        "The Berlin Wall fell in 1989, signalling the end of the Cold War.",
        "The British Empire at its height covered about a quarter of the Earth's land surface.",
        "The Enlightenment emphasized reason, science, and individual rights over tradition.",
        "The American Civil War ended slavery and transformed the federal government's power.",
        "World War Two killed an estimated 70 to 85 million people across all theatres of conflict.",
        "The Ottoman Empire lasted over six centuries before dissolving after World War One.",
        "Ancient Greek philosophy laid foundations for Western science, ethics, and politics.",
        "The Cold War divided the world into two ideological blocs for nearly half a century.",
        "The abolition of the transatlantic slave trade followed decades of activist campaigning.",
        "Alexander the Great spread Hellenistic culture across a vast territory by age 30.",
        "The Meiji Restoration rapidly modernized Japan by adopting Western technology.",
        "Gandhi's nonviolent resistance movement led India to independence from British rule.",
        "The Holocaust was the systematic genocide of six million Jews by Nazi Germany.",
        "The Cuban Missile Crisis brought the world to the brink of nuclear war in 1962.",
        "Ancient Mesopotamia developed the earliest writing systems, laws, and urban centres.",
        "The Magna Carta of 1215 established the principle that rulers are subject to law.",
        "The space race between the US and USSR culminated in the Apollo 11 moon landing.",
        "Decolonization after World War Two transformed dozens of nations across Africa and Asia.",
        "The Arab Spring uprisings beginning in 2010 reshaped governments across the Middle East.",
    ],
}


class DataLoader:
  

    def __init__(self, config: dict) -> None:
        self.config = config
        self.input_path = config["input_path"]
        self.max_docs = config.get("max_docs")
        self.min_doc_length = config.get("min_doc_length", 20)
        self.strip_html = config.get("strip_html", True)


    def load(self) -> List[str]:
       
        if self.input_path == "demo":
            documents = self._load_demo()
       
        else:
            path = Path(self.input_path)
            if not path.exists():
                logger.warning(
                    "Input path not found: %s  →  falling back to demo corpus", path
                )
                documents = self._load_demo()
            elif path.is_dir():
                documents = self._load_directory(path)
            elif path.suffix.lower() == ".csv":
                documents = self._load_csv(path)
            else:
                documents = self._load_txt(path)

        documents = self._clean(documents)

        
        if self.max_docs and len(documents) > self.max_docs:
            logger.info("Capping corpus at %d documents", self.max_docs)
            documents = documents[: self.max_docs]

        return documents


    def _load_demo(self) -> List[str]:

        logger.info("Using built-in demo corpus (300 documents, 10 topics)")
        docs = []
        
        for topic_docs in DEMO_CORPUS.values():
            docs.extend(topic_docs)
        return docs

    def _load_txt(self, path: Path) -> List[str]:

        logger.info("Loading text file: %s", path)
        
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return [line.strip() for line in lines]

    def _load_directory(self, directory: Path) -> List[str]:
        
        txt_files = sorted(directory.glob("*.txt"))
        logger.info("Loading %d .txt files from directory: %s", len(txt_files), directory)
        documents = []
        
        for fp in txt_files:
            text = fp.read_text(encoding="utf-8", errors="replace").strip()
            documents.append(text)
        return documents

    def _load_csv(self, path: Path) -> List[str]:
        
        logger.info("Loading CSV file: %s", path)
        preferred_cols = {"text", "content", "body", "abstract", "description"}
        documents = []
       
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            col = None
            for preferred in preferred_cols:
                if preferred in (reader.fieldnames or []):
                    col = preferred
                    break
            if col is None and reader.fieldnames:
                col = reader.fieldnames[0]
                logger.warning("No preferred text column found — using: %s", col)
            for row in reader:
                if col and col in row:
                    documents.append(row[col])
        return documents


    def _clean(self, documents: List[str]) -> List[str]:
        
        cleaned = []
        dropped = 0
        for doc in documents:
            doc = str(doc)                         
            doc = html.unescape(doc)            
            if self.strip_html:
                doc = re.sub(r"<[^>]+>", " ", doc) 
            doc = re.sub(r"\s+", " ", doc).strip()  

            if len(doc) < self.min_doc_length:
                dropped += 1
                continue
            cleaned.append(doc)

        if dropped > 0:
            logger.debug("Dropped %d documents shorter than %d chars", dropped, self.min_doc_length)

        return cleaned