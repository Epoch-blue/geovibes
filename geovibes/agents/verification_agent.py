"""LangGraph-based verification agent for validating detections."""

import os
from typing import Optional, TypedDict

from geovibes.agents.clustering import create_single_cluster
from geovibes.agents.places import PlacesClient, format_places_for_prompt
from geovibes.agents.schemas import (
    ClusterInfo,
    PlaceInfo,
    VerificationDecision,
    VerificationResult,
    VerifiedFacility,
)
from geovibes.agents.thumbnail import (
    generate_cluster_thumbnail,
    thumbnail_to_base64,
    load_reference_image,
)

try:
    from google import genai
    from google.genai import types as genai_types
    import base64

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GENAI_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = None
    END = None
    LANGGRAPH_AVAILABLE = False


# Pricing per million tokens (USD) - from https://ai.google.dev/gemini-api/docs/pricing
MODEL_PRICING = {
    # Gemini 3 series
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    # Gemini 2.5 series
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    # Gemini 2.0 series
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
}


DEFAULT_TARGET_PROMPT = """You are analyzing satellite imagery to identify palm oil processing mills.

Palm oil mills typically have these characteristics:
- Large industrial buildings/warehouses with metal roofs
- Tall smokestacks or chimneys (often visible as shadows)
- Settling ponds (rectangular or circular water features, often brown/orange colored)
- Storage tanks (cylindrical structures)
- Truck loading areas and access roads
- Located in rural/agricultural areas surrounded by palm oil plantations
- Processing equipment and conveyors visible

The facility should show clear industrial infrastructure consistent with palm oil processing,
not just agricultural land or other types of facilities."""


class AgentState(TypedDict):
    """State passed between agent nodes."""

    cluster: ClusterInfo
    target_prompt: str
    reference_image_b64: Optional[str]
    satellite_image_b64: str
    initial_decision: Optional[VerificationDecision]
    initial_confidence: float
    initial_reasoning: str
    observed_features: list[str]
    facility_type_guess: Optional[str]
    places_context: list[PlaceInfo]
    context_gathered: bool
    final_decision: Optional[VerificationDecision]
    facility: Optional[VerifiedFacility]
    rejection_reason: Optional[str]
    alternative_facility_type: Optional[str]


class VerificationAgent:
    """Agent for verifying detection clusters."""

    def __init__(
        self,
        target_type: str = "palm_oil_mill",
        target_prompt: Optional[str] = None,
        reference_image_path: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        google_maps_api_key: Optional[str] = None,
        confidence_threshold: float = 0.7,
        enable_places: bool = True,
        verbose: bool = False,
        model_name: str = "gemini-3-flash-preview",
        enable_search: bool = False,
    ):
        """
        Initialize the verification agent.

        Parameters
        ----------
        target_type : str
            Type of facility to verify (for logging/output)
        target_prompt : str, optional
            Custom prompt describing the target facility
        reference_image_path : str, optional
            Path to reference image of target facility type
        gemini_api_key : str, optional
            Gemini API key (falls back to GEMINI_API_KEY env var)
        google_maps_api_key : str, optional
            Google Maps API key (falls back to GOOGLE_MAPS_API_KEY env var)
        confidence_threshold : float
            Minimum confidence to accept without additional context
        enable_places : bool
            Whether to use Google Places API for context
        verbose : bool
            Print prompts and responses during execution
        model_name : str
            Gemini model to use (e.g., "gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite")
        enable_search : bool
            Enable Google Search grounding for enrichment queries
        """
        self.target_type = target_type
        self.target_prompt = target_prompt or DEFAULT_TARGET_PROMPT
        self.confidence_threshold = confidence_threshold
        self.enable_places = enable_places
        self.verbose = verbose
        self.model_name = model_name
        self.enable_search = enable_search

        self._reset_usage_tracking()

        self.reference_image_b64 = None
        if reference_image_path:
            try:
                self.reference_image_b64 = load_reference_image(reference_image_path)
            except FileNotFoundError:
                pass

        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.genai_client = None
        if GENAI_AVAILABLE:
            if self.gemini_api_key:
                self.genai_client = genai.Client(api_key=self.gemini_api_key)
            else:
                try:
                    self.genai_client = genai.Client()
                except ValueError:
                    self.genai_client = None

        if enable_places:
            self.places_client = PlacesClient(api_key=google_maps_api_key)
        else:
            self.places_client = None

        self._graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    def _reset_usage_tracking(self):
        """Reset token and cost tracking for a new verification."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.api_calls = 0

    def _log_usage(self, response, step_name: str):
        """Log token usage and cost from an API response."""
        if not hasattr(response, "usage_metadata") or response.usage_metadata is None:
            return

        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        pricing = MODEL_PRICING.get(self.model_name, {"input": 0.10, "output": 0.40})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        step_cost = input_cost + output_cost

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += step_cost
        self.api_calls += 1

        if self.verbose:
            print(f"\n[USAGE] {step_name}")
            print(
                f"  Tokens: {input_tokens:,} input + {output_tokens:,} output = {input_tokens + output_tokens:,} total"
            )
            print(
                f"  Cost: ${step_cost:.6f} (${input_cost:.6f} input + ${output_cost:.6f} output)"
            )
            print(
                f"  Running total: ${self.total_cost_usd:.6f} ({self.api_calls} API calls)"
            )

    def get_usage_summary(self) -> dict:
        """Get a summary of token usage and costs for the last verification."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "api_calls": self.api_calls,
            "model": self.model_name,
        }

    def _build_graph(self) -> "StateGraph":
        """Build the LangGraph state machine."""
        graph = StateGraph(AgentState)

        graph.add_node("initial_verification", self._initial_verification_node)
        graph.add_node("gather_context", self._gather_context_node)
        graph.add_node("second_verification", self._second_verification_node)
        graph.add_node("enrich_facility", self._enrich_facility_node)
        graph.add_node("reject_cluster", self._reject_cluster_node)

        graph.set_entry_point("initial_verification")

        graph.add_conditional_edges(
            "initial_verification",
            self._route_after_initial,
            {
                "gather_context": "gather_context",
                "reject_cluster": "reject_cluster",
            },
        )

        graph.add_conditional_edges(
            "gather_context",
            self._route_after_context,
            {
                "second_verification": "second_verification",
                "enrich_facility": "enrich_facility",
            },
        )

        graph.add_conditional_edges(
            "second_verification",
            self._route_after_second,
            {
                "enrich_facility": "enrich_facility",
                "reject_cluster": "reject_cluster",
            },
        )

        graph.add_edge("enrich_facility", END)
        graph.add_edge("reject_cluster", END)

        return graph.compile()

    def _route_after_initial(self, state: AgentState) -> str:
        """Route after initial verification."""
        decision = state.get("initial_decision")
        confidence = state.get("initial_confidence", 0)

        if (
            decision == VerificationDecision.INVALID
            and confidence >= self.confidence_threshold
        ):
            return "reject_cluster"
        return "gather_context"

    def _route_after_context(self, state: AgentState) -> str:
        """Route after gathering context - skip second verification if already confident."""
        decision = state.get("initial_decision")
        confidence = state.get("initial_confidence", 0)

        if (
            decision == VerificationDecision.VALID
            and confidence >= self.confidence_threshold
        ):
            return "enrich_facility"
        return "second_verification"

    def _route_after_second(self, state: AgentState) -> str:
        """Route after second verification with context."""
        decision = state.get("final_decision")
        if decision == VerificationDecision.VALID:
            return "enrich_facility"
        return "reject_cluster"

    def _initial_verification_node(self, state: AgentState) -> dict:
        """Perform initial verification using satellite imagery."""
        if self.verbose:
            print("\n" + "=" * 50)
            print("=== INITIAL VERIFICATION ===")
            print("=" * 50)

        if not self.genai_client:
            if self.verbose:
                print("[INFO] Model not available")
            return {
                "initial_decision": VerificationDecision.NEEDS_CONTEXT,
                "initial_confidence": 0.0,
                "initial_reasoning": "Model not available",
                "observed_features": [],
                "facility_type_guess": None,
            }

        cluster = state["cluster"]
        satellite_b64 = state["satellite_image_b64"]

        prompt = f"""{self.target_prompt}

Analyze the satellite image provided and determine if this location contains the target facility type.

Respond with a JSON object containing:
- decision: "valid" if this appears to be the target facility, "invalid" if clearly not, "needs_context" if uncertain
- confidence: a number between 0 and 1 indicating your confidence
- reasoning: brief explanation of your decision
- observed_features: list of features you observed in the image
- facility_type_guess: your best guess at what type of facility this is (if any)

Location: {cluster.centroid_lat:.6f}, {cluster.centroid_lon:.6f}
Detection count in cluster: {cluster.detection_count}
Average model probability: {cluster.avg_probability:.2f}

Respond ONLY with valid JSON."""

        if self.verbose:
            print("\n[PROMPT]")
            print(prompt)

        try:
            image_bytes = base64.b64decode(satellite_b64)
            image_part = genai_types.Part.from_bytes(
                data=image_bytes, mime_type="image/png"
            )

            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image_part],
                config={"response_mime_type": "application/json"},
            )

            self._log_usage(response, "Initial Verification")

            import json

            result = json.loads(response.text)

            decision_str = result.get("decision", "needs_context").lower()
            if decision_str == "valid":
                decision = VerificationDecision.VALID
            elif decision_str == "invalid":
                decision = VerificationDecision.INVALID
            else:
                decision = VerificationDecision.NEEDS_CONTEXT

            if self.verbose:
                print("\n[RESPONSE]")
                print(f"Decision: {decision.value}")
                print(f"Confidence: {result.get('confidence', 0.5):.2f}")
                print(f"Reasoning: {result.get('reasoning', '')}")
                print(f"Observed features: {result.get('observed_features', [])}")
                print(f"Facility type guess: {result.get('facility_type_guess')}")

            return {
                "initial_decision": decision,
                "initial_confidence": float(result.get("confidence", 0.5)),
                "initial_reasoning": result.get("reasoning", ""),
                "observed_features": result.get("observed_features", []),
                "facility_type_guess": result.get("facility_type_guess"),
            }

        except Exception as e:
            if self.verbose:
                print(f"\n[ERROR] {str(e)}")
            return {
                "initial_decision": VerificationDecision.NEEDS_CONTEXT,
                "initial_confidence": 0.0,
                "initial_reasoning": f"Error during verification: {str(e)}",
                "observed_features": [],
                "facility_type_guess": None,
            }

    def _gather_context_node(self, state: AgentState) -> dict:
        """Gather additional context from Google Places."""
        if self.verbose:
            print("\n" + "=" * 50)
            print("=== GATHERING CONTEXT ===")
            print("=" * 50)

        cluster = state["cluster"]
        places = []

        if self.places_client and self.places_client.is_available:
            places = self.places_client.search_nearby(
                lat=cluster.centroid_lat,
                lon=cluster.centroid_lon,
                radius_m=1000,
                limit=10,
            )
            if self.verbose:
                print(f"\n[PLACES API] Found {len(places)} nearby places:")
                for place in places:
                    print(f"  - {place.name} ({place.distance_m:.0f}m)")
        elif self.verbose:
            print("\n[INFO] Places API not available")

        return {
            "places_context": places,
            "context_gathered": True,
        }

    def _second_verification_node(self, state: AgentState) -> dict:
        """Re-verify with additional context."""
        if self.verbose:
            print("\n" + "=" * 50)
            print("=== SECOND VERIFICATION (with context) ===")
            print("=" * 50)

        if not self.genai_client:
            if self.verbose:
                print("[INFO] Model not available")
            initial = state.get("initial_decision", VerificationDecision.INVALID)
            return {"final_decision": initial}

        cluster = state["cluster"]
        satellite_b64 = state["satellite_image_b64"]
        places = state.get("places_context", [])
        initial_reasoning = state.get("initial_reasoning", "")
        observed_features = state.get("observed_features", [])

        places_text = format_places_for_prompt(places)

        prompt = f"""{self.target_prompt}

You are re-evaluating this location with additional context.

Previous analysis:
- Initial reasoning: {initial_reasoning}
- Observed features: {", ".join(observed_features) if observed_features else "None noted"}

{places_text}

Based on the satellite image AND the nearby places information, make a final determination.
Look for place names that might indicate the facility type (e.g., "PT XYZ Palm Oil Mill", "ABC Processing").

Respond with a JSON object containing:
- decision: "valid" if this is the target facility, "invalid" if not
- confidence: a number between 0 and 1
- reasoning: explanation incorporating the places context
- facility_type: what type of facility this actually is

Location: {cluster.centroid_lat:.6f}, {cluster.centroid_lon:.6f}

Respond ONLY with valid JSON."""

        if self.verbose:
            print("\n[PROMPT]")
            print(prompt)

        try:
            image_bytes = base64.b64decode(satellite_b64)
            image_part = genai_types.Part.from_bytes(
                data=image_bytes, mime_type="image/png"
            )

            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image_part],
                config={"response_mime_type": "application/json"},
            )

            self._log_usage(response, "Second Verification")

            import json

            result = json.loads(response.text)

            decision_str = result.get("decision", "invalid").lower()
            if decision_str == "valid":
                decision = VerificationDecision.VALID
            else:
                decision = VerificationDecision.INVALID

            if self.verbose:
                print("\n[RESPONSE]")
                print(f"Decision: {decision.value}")
                print(f"Confidence: {result.get('confidence', 0.5):.2f}")
                print(f"Reasoning: {result.get('reasoning', '')}")
                print(f"Facility type: {result.get('facility_type')}")

            return {
                "final_decision": decision,
                "initial_confidence": float(result.get("confidence", 0.5)),
                "initial_reasoning": result.get("reasoning", ""),
                "alternative_facility_type": result.get("facility_type"),
            }

        except Exception as e:
            if self.verbose:
                print(f"\n[ERROR] {str(e)}")
            return {
                "final_decision": state.get(
                    "initial_decision", VerificationDecision.INVALID
                ),
            }

    def _enrich_facility_node(self, state: AgentState) -> dict:
        """Extract facility details for valid detections."""
        if self.verbose:
            print("\n" + "=" * 50)
            print("=== ENRICHING FACILITY DETAILS ===")
            print("=" * 50)

        if not self.genai_client:
            if self.verbose:
                print("[INFO] Model not available, using basic facility info")
            cluster = state["cluster"]
            return {
                "facility": VerifiedFacility(
                    facility_type=self.target_type,
                    lat=cluster.centroid_lat,
                    lon=cluster.centroid_lon,
                    confidence=state.get("initial_confidence", 0.5),
                ),
                "final_decision": VerificationDecision.VALID,
            }

        cluster = state["cluster"]
        places = state.get("places_context", [])
        observed_features = state.get("observed_features", [])

        places_text = format_places_for_prompt(places)

        prompt = f"""Based on the analysis of this location, extract facility details.

Location: {cluster.centroid_lat:.6f}, {cluster.centroid_lon:.6f}
Observed features: {", ".join(observed_features) if observed_features else "Industrial facility"}

{places_text}

Extract the following information if available:
- company_name: Name of the company operating this facility (look for "PT", "Ltd", company names in nearby places)
- facility_name: Name of the specific facility
- address: Best available address
- facility_type: Type of facility (e.g., "palm oil mill", "processing plant")
- confidence: Your confidence in this identification (0-1)
- evidence: List of evidence supporting this identification

Respond with a JSON object containing these fields.
Respond ONLY with valid JSON."""

        if self.verbose:
            print("\n[PROMPT]")
            print(prompt)
            if self.enable_search:
                print("\n[SEARCH GROUNDING ENABLED]")

        try:
            config = {"response_mime_type": "application/json"}
            if self.enable_search and GENAI_AVAILABLE:
                config["tools"] = [
                    genai_types.Tool(google_search=genai_types.GoogleSearch())
                ]

            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=config,
            )

            self._log_usage(response, "Facility Enrichment")

            import json

            result = json.loads(response.text)

            facility = VerifiedFacility(
                company_name=result.get("company_name"),
                facility_name=result.get("facility_name"),
                facility_type=result.get("facility_type", self.target_type),
                address=result.get("address"),
                lat=cluster.centroid_lat,
                lon=cluster.centroid_lon,
                confidence=float(result.get("confidence", 0.7)),
                notes="; ".join(result.get("evidence", []))
                if result.get("evidence")
                else None,
            )

            if self.verbose:
                print("\n[RESPONSE]")
                print(f"Company name: {facility.company_name}")
                print(f"Facility name: {facility.facility_name}")
                print(f"Facility type: {facility.facility_type}")
                print(f"Address: {facility.address}")
                print(f"Confidence: {facility.confidence:.2f}")
                print(f"Evidence: {result.get('evidence', [])}")

            return {
                "facility": facility,
                "final_decision": VerificationDecision.VALID,
            }

        except Exception as e:
            if self.verbose:
                print(f"\n[ERROR] {str(e)}")
            return {
                "facility": VerifiedFacility(
                    facility_type=self.target_type,
                    lat=cluster.centroid_lat,
                    lon=cluster.centroid_lon,
                    confidence=state.get("initial_confidence", 0.5),
                ),
                "final_decision": VerificationDecision.VALID,
            }

    def _reject_cluster_node(self, state: AgentState) -> dict:
        """Record rejection details."""
        if self.verbose:
            print("\n" + "=" * 50)
            print("=== REJECTING CLUSTER ===")
            print("=" * 50)
            print(
                f"Reason: {state.get('initial_reasoning', 'Did not match target facility type')}"
            )
            alt_type = state.get("facility_type_guess") or state.get(
                "alternative_facility_type"
            )
            if alt_type:
                print(f"Alternative facility type: {alt_type}")

        return {
            "final_decision": VerificationDecision.INVALID,
            "rejection_reason": state.get(
                "initial_reasoning", "Did not match target facility type"
            ),
            "alternative_facility_type": state.get("facility_type_guess")
            or state.get("alternative_facility_type"),
        }

    def verify_cluster(self, cluster: ClusterInfo) -> VerificationResult:
        """
        Verify a detection cluster.

        Parameters
        ----------
        cluster : ClusterInfo
            Cluster to verify

        Returns
        -------
        VerificationResult
            Verification result with decision and details
        """
        self._reset_usage_tracking()

        try:
            thumbnail_bytes = generate_cluster_thumbnail(cluster)
            satellite_b64 = thumbnail_to_base64(thumbnail_bytes)
        except Exception:
            return VerificationResult(
                cluster=cluster,
                decision=VerificationDecision.INVALID,
                rejection_reason="Failed to generate satellite thumbnail",
            )

        initial_state: AgentState = {
            "cluster": cluster,
            "target_prompt": self.target_prompt,
            "reference_image_b64": self.reference_image_b64,
            "satellite_image_b64": satellite_b64,
            "initial_decision": None,
            "initial_confidence": 0.0,
            "initial_reasoning": "",
            "observed_features": [],
            "facility_type_guess": None,
            "places_context": [],
            "context_gathered": False,
            "final_decision": None,
            "facility": None,
            "rejection_reason": None,
            "alternative_facility_type": None,
        }

        if self._graph:
            final_state = self._graph.invoke(initial_state)
        else:
            final_state = self._run_without_langgraph(initial_state)

        if self.verbose and self.api_calls > 0:
            print("\n" + "=" * 50)
            print("=== COST SUMMARY ===")
            print("=" * 50)
            print(f"Model: {self.model_name}")
            print(f"API calls: {self.api_calls}")
            print(
                f"Total tokens: {self.total_input_tokens:,} input + {self.total_output_tokens:,} output"
            )
            print(f"Total cost: ${self.total_cost_usd:.6f}")

        return VerificationResult(
            cluster=cluster,
            decision=final_state.get("final_decision", VerificationDecision.INVALID),
            facility=final_state.get("facility"),
            rejection_reason=final_state.get("rejection_reason"),
            alternative_facility_type=final_state.get("alternative_facility_type"),
            places_found=final_state.get("places_context", []),
            context_gathered=final_state.get("context_gathered", False),
        )

    def _run_without_langgraph(self, state: AgentState) -> dict:
        """Fallback execution without LangGraph."""
        state = {**state, **self._initial_verification_node(state)}

        route = self._route_after_initial(state)

        if route == "enrich_facility":
            state = {**state, **self._enrich_facility_node(state)}
        elif route == "gather_context":
            state = {**state, **self._gather_context_node(state)}
            state = {**state, **self._second_verification_node(state)}

            route2 = self._route_after_second(state)
            if route2 == "enrich_facility":
                state = {**state, **self._enrich_facility_node(state)}
            else:
                state = {**state, **self._reject_cluster_node(state)}
        else:
            state = {**state, **self._reject_cluster_node(state)}

        return state

    def verify_location(
        self,
        lat: float,
        lon: float,
        probability: float = 0.5,
    ) -> VerificationResult:
        """
        Verify a single location (convenience method for testing).

        Parameters
        ----------
        lat : float
            Latitude
        lon : float
            Longitude
        probability : float
            Detection probability to assign

        Returns
        -------
        VerificationResult
            Verification result
        """
        cluster = create_single_cluster(lat, lon, probability)
        return self.verify_cluster(cluster)
