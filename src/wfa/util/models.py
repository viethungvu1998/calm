from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, JSON, Boolean, List
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Location(Base):
    __tablename__ = "locations"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    country = Column(String)
    region = Column(String)
    county = Column(String)
    locality = Column(String)
    time_zone = Column(String)
    crs = Column(String)

class URL(Base):
    __tablename__ = "urls"
    id = Column(Integer, primary_key=True)
    url = Column(String)
    source_type = Column(String)
    relevance = Column(Boolean)

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    location_id = Column(Integer, ForeignKey("locations.id"))
    location = relationship("Location")

    event_type = Column(String)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    area_burned_ha = Column(Float)
    intensity_level = Column(String)
    fuel_type = Column(String)
    weather_data = Column(JSON)
    insight = Column(String) 
    cause = Column(String)
    url_id = Column(Integer, ForeignKey("urls.id"))
    url = relationship("URL")

class ScholarlyInsight(Base):
    __tablename__ = "scholarly_insights"
    id = Column(Integer, primary_key=True)
    event = relationship("Event")
    insight_type = Column(String)
    insight_text = Column(String)
    source_url = Column(String)
    url_id = Column(Integer, ForeignKey("urls.id"))
    url = relationship("URL")


class NewsInsight(Base):
    __tablename__ = "news_insights"
    id = Column(Integer, primary_key=True)
    event = relationship("Event")
    insight_type = Column(String)
    insight_text = Column(String)
    fuel_type = Column(String)
    weather_data = Column(JSON)
    source_url = Column(String)
    url_id = Column(Integer, ForeignKey("urls.id"))
    url = relationship("URL")

