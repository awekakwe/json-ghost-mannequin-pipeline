#!/usr/bin/env python3
"""
step6_feedback_engine.py – Step 6/6 Feedback & Auto-Tuning
===========================================================

Persist telemetry, compute health, alert, and propose auto-tunes.

Data:
- Run id, SKU, variant, route, template id, QA metrics, ΔE, pass/fail, latency
- SQLite (dev) / Postgres (prod)

Logic:
- Composite health score; trend analysis; rules (YAML) to adjust seeds/guidance/route choice

Outputs:
- step6/feedback.db (or Postgres)
- step6/summary.json (optional), alerts (Slack/email in prod)

Dependencies: SQLAlchemy psycopg2-binary pyyaml requests
"""

from __future__ import annotations

import os
import json
import time
import uuid
import argparse
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import yaml

# Database dependencies
try:
    from sqlalchemy import (
        create_engine, Column, String, Float, Integer, Boolean, DateTime, Text,
        ForeignKey, JSON
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    from sqlalchemy.sql import func
    HAVE_SQLALCHEMY = True
    Base = declarative_base()
except ImportError:
    HAVE_SQLALCHEMY = False
    Base = None

# Optional notification dependencies
try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

logger = logging.getLogger("photostudio.feedback_engine")

# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------

if HAVE_SQLALCHEMY:
    class ProcessingRun(Base):
        """Main table for processing run telemetry."""
        __tablename__ = 'processing_runs'
        
        id = Column(String, primary_key=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # Product info
        sku = Column(String, nullable=False)
        variant = Column(String, nullable=False)
        
        # Pipeline info
        pipeline_version = Column(String, nullable=False)
        route = Column(String, nullable=False)  # "A" or "B"
        template_id = Column(String, nullable=False)
        
        # Quality metrics
        delta_e_mean = Column(Float)
        delta_e_p95 = Column(Float)
        hollow_quality = Column(Float)
        sharpness = Column(Float)
        background_score = Column(Float)
        composite_score = Column(Float)
        
        # Pass/fail gates
        passes_color_gate = Column(Boolean)
        passes_hollow_gate = Column(Boolean)
        passes_sharpness_gate = Column(Boolean)
        passes_all_gates = Column(Boolean)
        
        # Performance
        processing_time_seconds = Column(Float)
        candidates_generated = Column(Integer)
        
        # Settings used
        render_settings = Column(JSON)  # Store render configuration as JSON
        
        # Relationships
        tuning_events = relationship("TuningEvent", back_populates="run")

    class TuningEvent(Base):
        """Table for auto-tuning events and recommendations."""
        __tablename__ = 'tuning_events'
        
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # Associated run
        run_id = Column(String, ForeignKey('processing_runs.id'))
        run = relationship("ProcessingRun", back_populates="tuning_events")
        
        # Tuning info
        trigger_reason = Column(String, nullable=False)  # "low_quality", "trend_decline", etc.
        recommendation_type = Column(String, nullable=False)  # "guidance_adjust", "route_switch", etc.
        recommendation_data = Column(JSON)  # Specific tuning parameters
        
        # Confidence and status
        confidence = Column(Float)  # 0.0 to 1.0
        applied = Column(Boolean, default=False)
        effective = Column(Boolean)  # Set after evaluation

    class HealthSummary(Base):
        """Aggregated health metrics by time period.""" 
        __tablename__ = 'health_summaries'
        
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # Time period
        period_start = Column(DateTime, nullable=False)
        period_end = Column(DateTime, nullable=False) 
        period_type = Column(String, nullable=False)  # "hour", "day", "week"
        
        # Aggregate metrics
        total_runs = Column(Integer)
        success_rate = Column(Float)  # Fraction passing all gates
        avg_composite_score = Column(Float)
        avg_delta_e = Column(Float)
        avg_processing_time = Column(Float)
        
        # Health score (0.0 to 1.0)
        health_score = Column(Float)

# ---------------------------------------------------------------------------
# Configuration & Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FeedbackConfig:
    """Configuration for feedback engine."""
    db_url: str = "sqlite:///step6/feedback.db"
    enable_alerts: bool = False
    enable_auto_tuning: bool = True
    health_check_interval_hours: int = 24
    min_runs_for_tuning: int = 10
    
    # Alert thresholds
    alert_health_threshold: float = 0.7
    alert_success_rate_threshold: float = 0.85
    
    # Tuning thresholds
    tuning_delta_e_threshold: float = 2.5
    tuning_success_rate_threshold: float = 0.8
    tuning_confidence_threshold: float = 0.7

@dataclass
class TuningRule:
    """Auto-tuning rule definition."""
    name: str
    trigger_condition: str  # "low_quality", "high_latency", etc.
    parameter: str  # "guidance_scale", "route", etc.
    adjustment: Union[float, str]  # Amount to adjust or new value
    confidence: float
    cooldown_hours: int = 24

@dataclass
class AlertConfig:
    """Alert notification configuration."""
    webhook_url: Optional[str] = None
    email_smtp_host: Optional[str] = None
    email_recipients: List[str] = None

# ---------------------------------------------------------------------------
# Database Manager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(self, db_url: str):
        if not HAVE_SQLALCHEMY:
            raise ImportError("SQLAlchemy not installed")
            
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database initialized: {db_url}")
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def store_run_data(
        self,
        run_id: str,
        sku: str,
        variant: str,
        route: str,
        template_id: str,
        qa_data: Dict[str, Any],
        manifest_data: Dict[str, Any],
        processing_time: float,
        render_settings: Dict[str, Any]
    ) -> ProcessingRun:
        """Store processing run telemetry."""
        
        with self.get_session() as session:
            # Extract QA metrics
            qa_metrics = qa_data.get("best_candidate", {})
            final_metrics = qa_metrics.get("final_metrics", {}) if "final_metrics" in qa_metrics else qa_metrics
            
            run = ProcessingRun(
                id=run_id,
                sku=sku,
                variant=variant,
                pipeline_version=manifest_data.get("processing_metadata", {}).get("pipeline_version", "1.1.0"),
                route=route,
                template_id=template_id,
                
                # Quality metrics
                delta_e_mean=final_metrics.get("mean_delta_e"),
                delta_e_p95=final_metrics.get("p95_delta_e"),
                hollow_quality=final_metrics.get("hollow_quality"),
                sharpness=final_metrics.get("sharpness"),
                background_score=final_metrics.get("background_score"),
                composite_score=final_metrics.get("composite_score"),
                
                # Pass/fail
                passes_color_gate=final_metrics.get("mean_delta_e", 999) <= 2.0 if final_metrics.get("mean_delta_e") else None,
                passes_hollow_gate=final_metrics.get("hollow_quality", 0) >= 0.4 if final_metrics.get("hollow_quality") else None,
                passes_sharpness_gate=final_metrics.get("sharpness", 0) >= 0.3 if final_metrics.get("sharpness") else None,
                passes_all_gates=final_metrics.get("passes_gates", False),
                
                # Performance
                processing_time_seconds=processing_time,
                candidates_generated=qa_data.get("candidates_processed", 1),
                
                # Settings
                render_settings=render_settings
            )
            
            session.add(run)
            session.commit()
            session.refresh(run)
            
            logger.info(f"Stored run data: {run_id}")
            return run

    def get_recent_runs(self, hours: int = 24, limit: int = 100) -> List[ProcessingRun]:
        """Get recent processing runs."""
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            return session.query(ProcessingRun).filter(
                ProcessingRun.created_at >= cutoff
            ).order_by(ProcessingRun.created_at.desc()).limit(limit).all()

    def get_health_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Compute current health metrics."""
        recent_runs = self.get_recent_runs(hours)
        
        if not recent_runs:
            return {
                "total_runs": 0,
                "health_score": 0.0,
                "success_rate": 0.0,
                "avg_composite_score": 0.0,
                "avg_delta_e": 0.0,
                "avg_processing_time": 0.0
            }
        
        # Compute aggregate metrics
        total_runs = len(recent_runs)
        successful_runs = sum(1 for r in recent_runs if r.passes_all_gates)
        success_rate = successful_runs / total_runs
        
        # Quality metrics (filter out None values)
        composite_scores = [r.composite_score for r in recent_runs if r.composite_score is not None]
        delta_es = [r.delta_e_mean for r in recent_runs if r.delta_e_mean is not None]
        processing_times = [r.processing_time_seconds for r in recent_runs if r.processing_time_seconds is not None]
        
        avg_composite_score = statistics.mean(composite_scores) if composite_scores else 0.0
        avg_delta_e = statistics.mean(delta_es) if delta_es else 0.0
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0
        
        # Compute health score (weighted combination)
        health_score = (
            success_rate * 0.4 +
            min(1.0, avg_composite_score) * 0.3 +
            max(0.0, 1.0 - avg_delta_e / 5.0) * 0.2 +  # Lower ΔE is better
            max(0.0, 1.0 - avg_processing_time / 60.0) * 0.1  # Lower time is better
        )
        
        return {
            "total_runs": total_runs,
            "health_score": round(health_score, 3),
            "success_rate": round(success_rate, 3),
            "avg_composite_score": round(avg_composite_score, 3),
            "avg_delta_e": round(avg_delta_e, 2),
            "avg_processing_time": round(avg_processing_time, 2)
        }

# ---------------------------------------------------------------------------
# Auto-Tuning Engine
# ---------------------------------------------------------------------------

class AutoTuningEngine:
    """Engine for analyzing performance and proposing parameter adjustments."""
    
    def __init__(self, config: FeedbackConfig, db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self.tuning_rules = self._load_default_rules()
    
    def _load_default_rules(self) -> List[TuningRule]:
        """Load default auto-tuning rules."""
        return [
            TuningRule(
                name="high_delta_e_guidance_increase",
                trigger_condition="high_delta_e",
                parameter="guidance_scale",
                adjustment=0.5,  # Increase by 0.5
                confidence=0.8
            ),
            TuningRule(
                name="low_hollow_quality_route_switch",
                trigger_condition="low_hollow_quality",
                parameter="route",
                adjustment="B",  # Switch to Route B
                confidence=0.6
            ),
            TuningRule(
                name="high_latency_route_switch",
                trigger_condition="high_latency",
                parameter="route",
                adjustment="B",  # Switch to faster Route B
                confidence=0.7
            ),
            TuningRule(
                name="low_success_rate_seed_rotation",
                trigger_condition="low_success_rate",
                parameter="seed",
                adjustment="rotate",  # Signal to rotate seeds
                confidence=0.5
            )
        ]
    
    def load_rules_from_yaml(self, rules_path: Path):
        """Load tuning rules from YAML file."""
        if not rules_path.exists():
            logger.warning(f"Tuning rules file not found: {rules_path}")
            return
            
        with open(rules_path, 'r') as f:
            rules_data = yaml.safe_load(f)
            
        self.tuning_rules = []
        for rule_data in rules_data.get("rules", []):
            rule = TuningRule(
                name=rule_data["name"],
                trigger_condition=rule_data["trigger_condition"],
                parameter=rule_data["parameter"],
                adjustment=rule_data["adjustment"],
                confidence=rule_data.get("confidence", 0.5),
                cooldown_hours=rule_data.get("cooldown_hours", 24)
            )
            self.tuning_rules.append(rule)
            
        logger.info(f"Loaded {len(self.tuning_rules)} tuning rules")
    
    def analyze_and_recommend(self, run_id: str) -> List[Dict[str, Any]]:
        """Analyze recent performance and generate recommendations."""
        recommendations = []
        
        # Get recent performance data
        recent_runs = self.db.get_recent_runs(hours=24)
        health_metrics = self.db.get_health_metrics(hours=24)
        
        if len(recent_runs) < self.config.min_runs_for_tuning:
            logger.info(f"Not enough runs for tuning analysis ({len(recent_runs)} < {self.config.min_runs_for_tuning})")
            return recommendations
        
        # Check each trigger condition
        triggers = self._identify_triggers(health_metrics, recent_runs)
        
        for trigger in triggers:
            # Find applicable rules
            applicable_rules = [r for r in self.tuning_rules if r.trigger_condition == trigger]
            
            for rule in applicable_rules:
                # Check confidence threshold
                if rule.confidence < self.config.tuning_confidence_threshold:
                    continue
                    
                # Check cooldown (avoid duplicate recent recommendations)
                if self._is_in_cooldown(rule, run_id):
                    continue
                
                recommendation = {
                    "rule_name": rule.name,
                    "trigger": trigger,
                    "parameter": rule.parameter,
                    "adjustment": rule.adjustment,
                    "confidence": rule.confidence,
                    "reason": self._get_trigger_reason(trigger, health_metrics)
                }
                
                recommendations.append(recommendation)
                
                # Store tuning event
                self._store_tuning_event(run_id, rule, trigger, recommendation)
        
        return recommendations
    
    def _identify_triggers(self, health_metrics: Dict[str, Any], recent_runs: List) -> List[str]:
        """Identify performance triggers based on metrics."""
        triggers = []
        
        # High delta E trigger
        if health_metrics["avg_delta_e"] > self.config.tuning_delta_e_threshold:
            triggers.append("high_delta_e")
        
        # Low success rate trigger  
        if health_metrics["success_rate"] < self.config.tuning_success_rate_threshold:
            triggers.append("low_success_rate")
        
        # High latency trigger (>30s average)
        if health_metrics["avg_processing_time"] > 30.0:
            triggers.append("high_latency")
        
        # Low hollow quality trigger
        hollow_scores = [r.hollow_quality for r in recent_runs if r.hollow_quality is not None]
        if hollow_scores and statistics.mean(hollow_scores) < 0.4:
            triggers.append("low_hollow_quality")
        
        # Low composite score trigger
        if health_metrics["avg_composite_score"] < 0.6:
            triggers.append("low_composite_score")
        
        return triggers
    
    def _is_in_cooldown(self, rule: TuningRule, run_id: str) -> bool:
        """Check if rule is in cooldown period."""
        cutoff = datetime.utcnow() - timedelta(hours=rule.cooldown_hours)
        
        with self.db.get_session() as session:
            recent_event = session.query(TuningEvent).filter(
                TuningEvent.trigger_reason == rule.trigger_condition,
                TuningEvent.created_at >= cutoff
            ).first()
            
            return recent_event is not None
    
    def _get_trigger_reason(self, trigger: str, health_metrics: Dict[str, Any]) -> str:
        """Get human-readable reason for trigger."""
        reasons = {
            "high_delta_e": f"Average ΔE ({health_metrics['avg_delta_e']:.2f}) exceeds threshold ({self.config.tuning_delta_e_threshold})",
            "low_success_rate": f"Success rate ({health_metrics['success_rate']:.2f}) below threshold ({self.config.tuning_success_rate_threshold})",
            "high_latency": f"Average processing time ({health_metrics['avg_processing_time']:.1f}s) is high",
            "low_hollow_quality": "Average hollow quality score is below acceptable threshold",
            "low_composite_score": f"Average composite score ({health_metrics['avg_composite_score']:.2f}) is below target"
        }
        return reasons.get(trigger, f"Triggered condition: {trigger}")
    
    def _store_tuning_event(self, run_id: str, rule: TuningRule, trigger: str, recommendation: Dict[str, Any]):
        """Store tuning event in database."""
        with self.db.get_session() as session:
            event = TuningEvent(
                run_id=run_id,
                trigger_reason=trigger,
                recommendation_type=rule.parameter,
                recommendation_data=recommendation,
                confidence=rule.confidence
            )
            session.add(event)
            session.commit()

# ---------------------------------------------------------------------------
# Alert System
# ---------------------------------------------------------------------------

class AlertSystem:
    """Send alerts when system health degrades."""
    
    def __init__(self, config: FeedbackConfig, alert_config: AlertConfig):
        self.config = config
        self.alert_config = alert_config
    
    def check_and_alert(self, health_metrics: Dict[str, Any], run_id: str):
        """Check health metrics and send alerts if needed."""
        alerts = []
        
        # Health score alert
        if health_metrics["health_score"] < self.config.alert_health_threshold:
            alerts.append({
                "severity": "warning",
                "message": f"System health score ({health_metrics['health_score']:.2f}) below threshold ({self.config.alert_health_threshold})",
                "metrics": health_metrics
            })
        
        # Success rate alert
        if health_metrics["success_rate"] < self.config.alert_success_rate_threshold:
            alerts.append({
                "severity": "error",
                "message": f"Success rate ({health_metrics['success_rate']:.2f}) below threshold ({self.config.alert_success_rate_threshold})", 
                "metrics": health_metrics
            })
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert, run_id)
    
    def _send_alert(self, alert: Dict[str, Any], run_id: str):
        """Send individual alert."""
        logger.warning(f"Alert triggered: {alert['message']}")
        
        # Webhook notification
        if self.alert_config.webhook_url and HAVE_REQUESTS:
            self._send_webhook_alert(alert, run_id)
    
    def _send_webhook_alert(self, alert: Dict[str, Any], run_id: str):
        """Send webhook alert (e.g., to Slack)."""
        payload = {
            "text": f"🚨 Photostudio Pipeline Alert",
            "attachments": [{
                "color": "warning" if alert["severity"] == "warning" else "danger",
                "fields": [
                    {
                        "title": "Message",
                        "value": alert["message"],
                        "short": False
                    },
                    {
                        "title": "Run ID", 
                        "value": run_id,
                        "short": True
                    },
                    {
                        "title": "Health Score",
                        "value": f"{alert['metrics']['health_score']:.2f}",
                        "short": True
                    }
                ]
            }]
        }
        
        try:
            response = requests.post(self.alert_config.webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Alert sent successfully")
            else:
                logger.error(f"Alert webhook failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

# ---------------------------------------------------------------------------
# Main Feedback Engine
# ---------------------------------------------------------------------------

class FeedbackEngine:
    """Main feedback and auto-tuning engine."""
    
    def __init__(
        self,
        config: FeedbackConfig,
        alert_config: Optional[AlertConfig] = None,
        tuning_rules_path: Optional[Path] = None
    ):
        self.config = config
        self.db = DatabaseManager(config.db_url)
        self.tuning_engine = AutoTuningEngine(config, self.db)
        self.alert_system = AlertSystem(config, alert_config) if alert_config else None
        
        # Load custom tuning rules if provided
        if tuning_rules_path:
            self.tuning_engine.load_rules_from_yaml(tuning_rules_path)
    
    def process_run_feedback(
        self,
        run_id: str,
        sku: str,
        variant: str,
        route: str,
        template_id: str,
        qa_report_path: Path,
        manifest_path: Path,
        processing_time: float,
        render_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process feedback for a completed pipeline run."""
        
        # Load input data
        qa_data = {}
        if qa_report_path.exists():
            with open(qa_report_path, 'r') as f:
                qa_data = json.load(f)
        
        manifest_data = {}
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
        
        render_settings = render_settings or {}
        
        # Store telemetry
        run = self.db.store_run_data(
            run_id=run_id,
            sku=sku,
            variant=variant,
            route=route,
            template_id=template_id,
            qa_data=qa_data,
            manifest_data=manifest_data,
            processing_time=processing_time,
            render_settings=render_settings
        )
        
        # Compute current health metrics
        health_metrics = self.db.get_health_metrics()
        
        # Check for alerts
        if self.alert_system and self.config.enable_alerts:
            self.alert_system.check_and_alert(health_metrics, run_id)
        
        # Generate tuning recommendations
        recommendations = []
        if self.config.enable_auto_tuning:
            recommendations = self.tuning_engine.analyze_and_recommend(run_id)
        
        # Prepare summary
        summary = {
            "run_id": run_id,
            "sku": f"{sku}-{variant}",
            "route": route,
            "template_id": template_id,
            "processing_time": processing_time,
            "health_metrics": health_metrics,
            "recommendations": recommendations,
            "alerts_triggered": len(recommendations) > 0,
            "timestamp": time.time()
        }
        
        logger.info(f"Feedback processed for run {run_id}")
        return summary

# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 6/6: Feedback & Auto-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--sku", required=True, help="Product SKU")
    parser.add_argument("--variant", required=True, help="Product variant")
    parser.add_argument("--route", required=True, choices=["A", "B"], help="Rendering route used")
    parser.add_argument("--template-id", required=True, help="Template ID used")
    parser.add_argument("--qa", required=True, help="Path to qa_report.json from Step 4")
    parser.add_argument("--manifest", required=True, help="Path to delivery_manifest.json from Step 5")
    parser.add_argument("--processing-time", type=float, required=True, help="Total processing time in seconds")
    parser.add_argument("--out", default="./step6", help="Output directory")
    
    # Configuration options
    parser.add_argument("--db", default="sqlite:///step6/feedback.db", help="Database URL")
    parser.add_argument("--enable-alerts", action="store_true", help="Enable health alerts")
    parser.add_argument("--enable-auto-tuning", action="store_true", default=True, help="Enable auto-tuning")
    parser.add_argument("--tuning-rules", help="Path to tuning rules YAML file")
    parser.add_argument("--webhook-url", help="Webhook URL for alerts (e.g., Slack)")

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate input files
    qa_path = Path(args.qa)
    manifest_path = Path(args.manifest)
    
    if not qa_path.exists():
        raise FileNotFoundError(f"QA report not found: {qa_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Delivery manifest not found: {manifest_path}")

    # Configuration
    feedback_config = FeedbackConfig(
        db_url=args.db,
        enable_alerts=args.enable_alerts,
        enable_auto_tuning=args.enable_auto_tuning
    )
    
    alert_config = None
    if args.webhook_url:
        alert_config = AlertConfig(webhook_url=args.webhook_url)
    
    tuning_rules_path = Path(args.tuning_rules) if args.tuning_rules else None

    # Initialize feedback engine
    engine = FeedbackEngine(feedback_config, alert_config, tuning_rules_path)

    try:
        # Process feedback
        summary = engine.process_run_feedback(
            run_id=args.run_id,
            sku=args.sku,
            variant=args.variant,
            route=args.route,
            template_id=args.template_id,
            qa_report_path=qa_path,
            manifest_path=manifest_path,
            processing_time=args.processing_time
        )

        # Save summary
        summary_path = out_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        # Print results
        print(f"✅ Feedback processing complete! Output saved to: {args.out}")
        print(f"🔍 Run: {args.run_id} ({args.sku}-{args.variant})")
        print(f"📊 Health Score: {summary['health_metrics']['health_score']:.3f}")
        print(f"✓ Success Rate: {summary['health_metrics']['success_rate']:.2f}")
        print(f"🎯 Avg ΔE: {summary['health_metrics']['avg_delta_e']:.2f}")
        print(f"⏱️  Processing Time: {args.processing_time:.1f}s")
        
        if summary['recommendations']:
            print(f"🔧 Auto-tuning recommendations: {len(summary['recommendations'])}")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"   {i}. {rec['rule_name']}: {rec['parameter']} → {rec['adjustment']} (confidence: {rec['confidence']:.2f})")
        else:
            print("🔧 No auto-tuning recommendations at this time")
            
        if summary['alerts_triggered']:
            print("🚨 Health alerts were triggered")

    except Exception as e:
        logger.error(f"Feedback processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
