"""
Y2AI SOCIAL PUBLISHER
Automated posting to Twitter/X, LinkedIn, and Bluesky

Replaces manual copy/paste workflow with scheduled automation.

Schedule (from your posting strategy):
- Daily: Stock tracker update at 4:45 PM ET
- Monday: Newsletter announcement at 8:30 AM ET
- Ad-hoc: Breaking news posts
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SocialPost:
    """A social media post"""
    id: Optional[str]
    platform: str  # twitter, linkedin, bluesky
    content: str
    thread: Optional[List[str]]  # For Twitter threads
    image_path: Optional[str]
    scheduled_at: Optional[str]
    posted_at: Optional[str]
    post_url: Optional[str]
    status: str  # draft, scheduled, posted, failed
    post_type: str  # daily_tracker, newsletter, breaking_news
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# PLATFORM ADAPTERS
# =============================================================================

class PlatformAdapter(ABC):
    """Base class for social media platform adapters"""
    
    @abstractmethod
    def post(self, content: str, image_path: Optional[str] = None) -> Optional[str]:
        """Post content and return post URL"""
        pass
    
    @abstractmethod
    def post_thread(self, tweets: List[str], image_path: Optional[str] = None) -> Optional[str]:
        """Post a thread and return first post URL"""
        pass
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        pass


class TwitterAdapter(PlatformAdapter):
    """Twitter/X API v2 integration"""
    
    def __init__(self):
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_secret = os.getenv('TWITTER_ACCESS_SECRET')
        self.client = None
        
        self._init_client()
    
    @property
    def platform_name(self) -> str:
        return "twitter"
    
    def _init_client(self):
        """Initialize Twitter client"""
        if not all([self.api_key, self.api_secret, self.access_token, self.access_secret]):
            logger.warning("Twitter API credentials not fully configured")
            return
        
        try:
            import tweepy
            
            self.client = tweepy.Client(
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_secret
            )
            logger.info("Twitter client initialized")
        except ImportError:
            logger.warning("tweepy not installed")
        except Exception as e:
            logger.error(f"Twitter client error: {e}")
    
    def post(self, content: str, image_path: Optional[str] = None) -> Optional[str]:
        """Post a single tweet"""
        if not self.client:
            logger.error("Twitter client not initialized")
            return None
        
        try:
            # TODO: Handle image upload if image_path provided
            response = self.client.create_tweet(text=content)
            
            if response.data:
                tweet_id = response.data['id']
                url = f"https://twitter.com/i/web/status/{tweet_id}"
                logger.info(f"Posted to Twitter: {url}")
                return url
                
        except Exception as e:
            logger.error(f"Twitter post error: {e}")
        
        return None
    
    def post_thread(self, tweets: List[str], image_path: Optional[str] = None) -> Optional[str]:
        """Post a Twitter thread"""
        if not self.client:
            logger.error("Twitter client not initialized")
            return None
        
        try:
            first_url = None
            reply_to_id = None
            
            for i, tweet_text in enumerate(tweets):
                # First tweet may have image
                if i == 0 and image_path:
                    # TODO: Upload image and attach
                    pass
                
                if reply_to_id:
                    response = self.client.create_tweet(
                        text=tweet_text,
                        in_reply_to_tweet_id=reply_to_id
                    )
                else:
                    response = self.client.create_tweet(text=tweet_text)
                
                if response.data:
                    reply_to_id = response.data['id']
                    if i == 0:
                        first_url = f"https://twitter.com/i/web/status/{reply_to_id}"
            
            logger.info(f"Posted Twitter thread: {first_url}")
            return first_url
            
        except Exception as e:
            logger.error(f"Twitter thread error: {e}")
        
        return None


class LinkedInAdapter(PlatformAdapter):
    """LinkedIn API integration"""
    
    def __init__(self):
        self.access_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
        self.person_urn = os.getenv('LINKEDIN_PERSON_URN')  # urn:li:person:xxxxx
        
    @property
    def platform_name(self) -> str:
        return "linkedin"
    
    def post(self, content: str, image_path: Optional[str] = None) -> Optional[str]:
        """Post to LinkedIn"""
        if not self.access_token or not self.person_urn:
            logger.warning("LinkedIn credentials not configured")
            return None
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0"
            }
            
            post_data = {
                "author": self.person_urn,
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {
                            "text": content
                        },
                        "shareMediaCategory": "NONE"
                    }
                },
                "visibility": {
                    "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                }
            }
            
            response = requests.post(
                "https://api.linkedin.com/v2/ugcPosts",
                headers=headers,
                json=post_data,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                post_urn = response.json().get('id', '')
                # LinkedIn doesn't easily give back a URL, construct it
                logger.info(f"Posted to LinkedIn: {post_urn}")
                return f"https://www.linkedin.com/feed/update/{post_urn}"
            else:
                logger.error(f"LinkedIn API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"LinkedIn post error: {e}")
        
        return None
    
    def post_thread(self, tweets: List[str], image_path: Optional[str] = None) -> Optional[str]:
        """LinkedIn doesn't support threads - post combined content"""
        combined = "\n\n".join(tweets)
        return self.post(combined, image_path)


class BlueskyAdapter(PlatformAdapter):
    """Bluesky AT Protocol integration"""
    
    def __init__(self):
        self.handle = os.getenv('BLUESKY_HANDLE')  # e.g., y2ai.bsky.social
        self.app_password = os.getenv('BLUESKY_APP_PASSWORD')
        self.session = None
        
    @property
    def platform_name(self) -> str:
        return "bluesky"
    
    def _create_session(self):
        """Create Bluesky session"""
        if not self.handle or not self.app_password:
            logger.warning("Bluesky credentials not configured")
            return False
        
        try:
            import requests
            
            response = requests.post(
                "https://bsky.social/xrpc/com.atproto.server.createSession",
                json={
                    "identifier": self.handle,
                    "password": self.app_password
                },
                timeout=30
            )
            
            if response.status_code == 200:
                self.session = response.json()
                logger.info("Bluesky session created")
                return True
            else:
                logger.error(f"Bluesky auth error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Bluesky session error: {e}")
        
        return False
    
    def post(self, content: str, image_path: Optional[str] = None) -> Optional[str]:
        """Post to Bluesky"""
        if not self.session:
            if not self._create_session():
                return None
        
        try:
            import requests
            
            # Bluesky has a 300 character limit
            if len(content) > 300:
                content = content[:297] + "..."
            
            post_data = {
                "repo": self.session['did'],
                "collection": "app.bsky.feed.post",
                "record": {
                    "$type": "app.bsky.feed.post",
                    "text": content,
                    "createdAt": datetime.utcnow().isoformat() + "Z"
                }
            }
            
            response = requests.post(
                "https://bsky.social/xrpc/com.atproto.repo.createRecord",
                headers={"Authorization": f"Bearer {self.session['accessJwt']}"},
                json=post_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                # Construct Bluesky URL
                uri = data.get('uri', '')
                # Extract rkey from uri like at://did:plc:xxx/app.bsky.feed.post/rkey
                rkey = uri.split('/')[-1] if uri else ''
                url = f"https://bsky.app/profile/{self.handle}/post/{rkey}"
                logger.info(f"Posted to Bluesky: {url}")
                return url
            else:
                logger.error(f"Bluesky post error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Bluesky post error: {e}")
        
        return None
    
    def post_thread(self, tweets: List[str], image_path: Optional[str] = None) -> Optional[str]:
        """Post a Bluesky thread (using replies)"""
        if not self.session:
            if not self._create_session():
                return None
        
        try:
            import requests
            
            first_url = None
            parent_ref = None
            root_ref = None
            
            for i, text in enumerate(tweets):
                # Truncate if needed
                if len(text) > 300:
                    text = text[:297] + "..."
                
                record = {
                    "$type": "app.bsky.feed.post",
                    "text": text,
                    "createdAt": datetime.utcnow().isoformat() + "Z"
                }
                
                # Add reply reference if not first post
                if parent_ref and root_ref:
                    record["reply"] = {
                        "root": root_ref,
                        "parent": parent_ref
                    }
                
                post_data = {
                    "repo": self.session['did'],
                    "collection": "app.bsky.feed.post",
                    "record": record
                }
                
                response = requests.post(
                    "https://bsky.social/xrpc/com.atproto.repo.createRecord",
                    headers={"Authorization": f"Bearer {self.session['accessJwt']}"},
                    json=post_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    uri = data.get('uri', '')
                    cid = data.get('cid', '')
                    
                    # Set up references for next post
                    current_ref = {"uri": uri, "cid": cid}
                    parent_ref = current_ref
                    
                    if i == 0:
                        root_ref = current_ref
                        rkey = uri.split('/')[-1]
                        first_url = f"https://bsky.app/profile/{self.handle}/post/{rkey}"
            
            logger.info(f"Posted Bluesky thread: {first_url}")
            return first_url
            
        except Exception as e:
            logger.error(f"Bluesky thread error: {e}")
        
        return None


# =============================================================================
# SOCIAL PUBLISHER
# =============================================================================

class SocialPublisher:
    """
    Main social publishing orchestrator
    
    Coordinates posting across Twitter, LinkedIn, and Bluesky.
    """
    
    def __init__(self):
        self.adapters = {
            "twitter": TwitterAdapter(),
            "linkedin": LinkedInAdapter(),
            "bluesky": BlueskyAdapter()
        }
    
    def publish_to_all(
        self,
        content: str,
        platforms: List[str] = None,
        thread: List[str] = None,
        image_path: Optional[str] = None,
        post_type: str = "general"
    ) -> Dict[str, Optional[str]]:
        """
        Publish content to multiple platforms
        
        Args:
            content: Main content (used for LinkedIn, single tweets)
            platforms: List of platforms to post to (default: all)
            thread: List of tweets for Twitter/Bluesky threads
            image_path: Path to image to attach
            post_type: Type of post for logging
        
        Returns:
            Dict mapping platform to post URL (or None if failed)
        """
        if platforms is None:
            platforms = ["twitter", "linkedin", "bluesky"]
        
        results = {}
        
        for platform in platforms:
            adapter = self.adapters.get(platform)
            if not adapter:
                logger.warning(f"Unknown platform: {platform}")
                continue
            
            try:
                if thread and platform in ["twitter", "bluesky"]:
                    url = adapter.post_thread(thread, image_path)
                else:
                    url = adapter.post(content, image_path)
                
                results[platform] = url
                
                if url:
                    logger.info(f"✅ Posted to {platform}: {url}")
                else:
                    logger.warning(f"❌ Failed to post to {platform}")
                    
            except Exception as e:
                logger.error(f"Error posting to {platform}: {e}")
                results[platform] = None
        
        return results
    
    def publish_daily_tracker(self, report_content: str) -> Dict[str, Optional[str]]:
        """
        Publish daily stock tracker update
        
        Posted to all platforms at 4:45 PM ET.
        """
        return self.publish_to_all(
            content=report_content,
            post_type="daily_tracker"
        )
    
    def publish_newsletter(
        self,
        linkedin_post: str,
        twitter_thread: List[str]
    ) -> Dict[str, Optional[str]]:
        """
        Publish newsletter announcement
        
        Posted Monday at 8:30 AM ET.
        """
        results = {}
        
        # LinkedIn gets the full post
        linkedin_result = self.adapters["linkedin"].post(linkedin_post)
        results["linkedin"] = linkedin_result
        
        # Twitter gets the thread
        twitter_result = self.adapters["twitter"].post_thread(twitter_thread)
        results["twitter"] = twitter_result
        
        # Bluesky gets the thread
        bluesky_result = self.adapters["bluesky"].post_thread(twitter_thread)
        results["bluesky"] = bluesky_result
        
        return results


# =============================================================================
# POST TEMPLATES
# =============================================================================

class PostTemplates:
    """Templates for common post types"""
    
    @staticmethod
    def daily_tracker_post(
        date: str,
        y2ai_today: float,
        spy_today: float,
        status: str,
        best_stock: str,
        best_change: float,
        worst_stock: str,
        worst_change: float,
        y2ai_ytd: float,
        spy_ytd: float
    ) -> str:
        """Generate daily tracker post"""
        
        emoji = {
            "VALIDATING": "✅",
            "NEUTRAL": "⚪",
            "CONTRADICTING": "⚠️",
            "STRONGLY VALIDATING": "🔥",
            "STRONGLY CONTRADICTING": "📉"
        }.get(status, "⚪")
        
        return f"""📊 Y2AI Infrastructure Index | {date}

Index: {y2ai_today:+.2f}%
S&P 500: {spy_today:+.2f}%

{emoji} {status}

Best: {best_stock} ({best_change:+.1f}%)
Worst: {worst_stock} ({worst_change:+.1f}%)

YTD: Y2AI {y2ai_ytd:+.1f}% | SPY {spy_ytd:+.1f}%

Dashboard: https://y2ai.us

#AI #Infrastructure #MarketData"""
    
    @staticmethod
    def newsletter_linkedin(
        edition_number: int,
        title: str,
        hook: str,
        key_findings: List[str],
        prediction: str,
        link: str
    ) -> str:
        """Generate LinkedIn newsletter announcement"""
        
        findings = "\n".join([f"→ {f}" for f in key_findings[:3]])
        
        return f"""📰 Y2AI Weekly Edition #{edition_number}: {title}

{hook}

Key findings:
{findings}

Prediction: {prediction}

Full analysis: {link}

#AI #Infrastructure #TechInvesting #Y2AI"""
    
    @staticmethod
    def newsletter_twitter_thread(
        edition_number: int,
        title: str,
        hook: str,
        key_points: List[str],
        prediction: str,
        link: str
    ) -> List[str]:
        """Generate Twitter thread for newsletter"""
        
        tweets = [
            f"📰 Y2AI Weekly #{edition_number}: {title}\n\n{hook}\n\n🧵",
        ]
        
        for i, point in enumerate(key_points[:4], 1):
            tweets.append(f"{i}/ {point}")
        
        tweets.append(f"Prediction: {prediction}")
        tweets.append(f"Full analysis: {link}\n\n#AI #Infrastructure #Y2AI")
        
        return tweets


# =============================================================================
# STORAGE INTEGRATION
# =============================================================================

def log_post(post: SocialPost):
    """Log post to Supabase"""
    try:
        from .storage import get_storage
        
        storage = get_storage()
        if hasattr(storage, 'client') and storage.is_connected():
            storage.client.table("social_posts").insert(
                post.to_dict()
            ).execute()
    except Exception as e:
        logger.error(f"Post logging error: {e}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    publisher = SocialPublisher()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test post
        test_content = "🧪 Test post from Y2AI automation system. Please ignore!"
        
        print("Testing social publisher...")
        print("(This will actually post if credentials are configured!)")
        
        # Only test one platform at a time for safety
        platform = sys.argv[2] if len(sys.argv) > 2 else "bluesky"
        
        results = publisher.publish_to_all(
            content=test_content,
            platforms=[platform]
        )
        
        print(f"\nResults: {results}")
    else:
        print("Y2AI Social Publisher")
        print("=====================")
        print()
        print("Usage:")
        print("  python -m y2ai.social_publisher --test [platform]")
        print()
        print("Platforms: twitter, linkedin, bluesky")
        print()
        print("Environment variables required:")
        print("  TWITTER_API_KEY, TWITTER_API_SECRET")
        print("  TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET")
        print("  LINKEDIN_ACCESS_TOKEN, LINKEDIN_PERSON_URN")
        print("  BLUESKY_HANDLE, BLUESKY_APP_PASSWORD")
