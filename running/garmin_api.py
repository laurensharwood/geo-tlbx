#!/usr/bin/ python

"""
in terminal: 
export EMAIL=your garmin username/email 
export PASSWORD=your garmin pwd
export POSTPWD=user/postgresql password
"""
import os, sys
import shutil
import numpy as np
import pandas as pd
import json
import logging
from getpass import getpass
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from withings_sync import fit
import readchar
import requests
import garth
import datetime
from datetime import datetime, timedelta, date, timezone
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Garmin:
    """
    from https://github.com/cyberjunky/python-garminconnect/blob/master/garminconnect/__init__.py
    Class for fetching data from Garmin Connect.
    """

    def __init__(
        self, email=None, password=None, is_cn=False, prompt_mfa=None
    ):
        """Create a new class instance."""
        self.username = email
        self.password = password
        self.is_cn = is_cn
        self.prompt_mfa = prompt_mfa
        self.garmin_connect_activities = ("/activitylist-service/activities/search/activities")
        self.garmin_connect_activity = "/activity-service/activity"
        self.garmin_connect_activity_types = ("/activity-service/activity/activityTypes" )
        self.garmin_connect_fit_download = "/download-service/files/activity"
        self.garmin_connect_tcx_download = ("/download-service/export/tcx/activity")
        self.garmin_connect_gpx_download = ("/download-service/export/gpx/activity")
        self.garmin_connect_kml_download = ("/download-service/export/kml/activity")
        self.garmin_connect_csv_download = ("/download-service/export/csv/activity")
        self.garmin_workouts = "/workout-service"
        self.garth = garth.Client(domain="garmin.cn" if is_cn else "garmin.com")

        self.display_name = None
        self.full_name = None
        self.unit_system = None

    def connectapi(self, path, **kwargs):
        return self.garth.connectapi(path, **kwargs)

    def download(self, path, **kwargs):
        return self.garth.download(path, **kwargs)

    def login(self, /, tokenstore: Optional[str] = None):
        """Log in using Garth."""
        tokenstore = tokenstore or os.getenv("GARMINTOKENS")

        if tokenstore:
            if len(tokenstore) > 512:
                self.garth.loads(tokenstore)
            else:
                self.garth.load(tokenstore)
        else:
            self.garth.login(
                self.username, self.password, prompt_mfa=self.prompt_mfa
            )

        self.display_name = self.garth.profile["displayName"]
        self.full_name = self.garth.profile["fullName"]

        return True


    def get_activities_by_date(self, startdate, enddate, activitytype=None):
        """
        Fetch available activities between specific dates
        :param startdate: String in the format YYYY-MM-DD
        :param enddate: String in the format YYYY-MM-DD
        :param activitytype: (Optional) Type of activity you are searching
                             Possible values are [cycling, biking]
        :return: list of JSON activities
        """

        activities = []
        start = 0
        limit = 20
        # mimicking the behavior of the web interface that fetches
        # 20 activities at a time
        # and automatically loads more on scroll
        url = self.garmin_connect_activities
        params = {
            "startDate": str(startdate),
            "endDate": str(enddate),
            "start": str(start),
            "limit": str(limit),
        }
        if activitytype:
            params["activityType"] = str(activitytype)

        logger.debug(
            f"Requesting activities by date from {startdate} to {enddate}"
        )
        while True:
            params["start"] = str(start)
            logger.debug(f"Requesting activities {start} to {start+limit}")
            act = self.connectapi(url, params=params)
            if act:
                activities.extend(act)
                start = start + limit
            else:
                break

        return activities

    class ActivityDownloadFormat(Enum):
        """Activity variables."""

        ORIGINAL = auto()
        TCX = auto()
        GPX = auto()
        KML = auto()
        CSV = auto()

    def download_activity(
        self, activity_id, dl_fmt=ActivityDownloadFormat.TCX
    ):
        """
        Downloads activity in requested format and returns the raw bytes. For
        "Original" will return the zip file content, up to user to extract it.
        "CSV" will return a csv of the splits.
        """
        activity_id = str(activity_id)
        urls = {
            Garmin.ActivityDownloadFormat.ORIGINAL: f"{self.garmin_connect_fit_download}/{activity_id}",  # noqa
            Garmin.ActivityDownloadFormat.TCX: f"{self.garmin_connect_tcx_download}/{activity_id}",  # noqa
            Garmin.ActivityDownloadFormat.GPX: f"{self.garmin_connect_gpx_download}/{activity_id}",  # noqa
            Garmin.ActivityDownloadFormat.KML: f"{self.garmin_connect_kml_download}/{activity_id}",  # noqa
            Garmin.ActivityDownloadFormat.CSV: f"{self.garmin_connect_csv_download}/{activity_id}",  # noqa
        }
        if dl_fmt not in urls:
            raise ValueError(f"Unexpected value {dl_fmt} for dl_fmt")
        url = urls[dl_fmt]

        logger.debug("Downloading activities from %s", url)

        return self.download(url)

class GarminConnectConnectionError(Exception):
    """Raised when communication ended in error."""

class GarminConnectTooManyRequestsError(Exception):
    """Raised when rate limit is exceeded."""

class GarminConnectAuthenticationError(Exception):
    """Raised when authentication is failed."""

class GarminConnectInvalidFileFormatError(Exception):
    """Raised when an invalid file format is passed to upload."""

def display_json(api_call, output):
    """Format API output for better readability."""

    dashed = "-" * 20
    header = f"{dashed} {api_call} {dashed}"
    footer = "-" * len(header)

    print(header)

    if isinstance(output, (int, str, dict, list)):
        print(json.dumps(output, indent=4))
    else:
        print(output)

    print(footer)

def display_text(output):
    """Format API output for better readability."""
    dashed = "-" * 60
    header = f"{dashed}"
    footer = "-" * len(header)

def get_credentials():
    """Get user credentials."""

    email = input("Login e-mail: ")
    password = getpass("Enter password: ")

    return email, password


def init_api(email, password, tokenstore):
    """Initialize Garmin API with your credentials."""
    try:
       ## print(f"Trying to login to Garmin Connect using token data from ...\n'{tokenstore}'" )
        garmin = Garmin()
        garmin.login(tokenstore)
    except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError):
        # Session is expired. You'll need to log in again
       ## print(  "Login tokens not present, login with your Garmin Connect credentials to generate them.\n"  f"They will be stored in '{tokenstore}' for future use.\n")
        try:
            # Ask for credentials if not set as environment variables
            if not email or not password:
                email, password = get_credentials()

            garmin = Garmin(email, password)
            garmin.login()
            # Save tokens for next login
            garmin.garth.dump(tokenstore)

        except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError, requests.exceptions.HTTPError) as err:
            logger.error(err)
            return None

    return garmin

######################################################################

def switch(api, option, out_dir, startdate, today):
    """Run selected API call."""

    if api:
        activitytype=""
        activities = api.get_activities_by_date(startdate.isoformat(), today.isoformat(), activitytype )
        for activity in activities:
            activity_id = activity["activityId"]
            activity_name = activity["activityName"]
            activity_start = activity["startTimeLocal"].replace(" ", "", -1).replace(":", "", -1).replace("-", "", -1)            
            if option==".gpx":
                gpx_data = api.download_activity( activity_id, dl_fmt=api.ActivityDownloadFormat.GPX)
                with open(os.path.join(out_dir, f"{str(activity_start)}.gpx"), "wb") as fb:
                    fb.write(gpx_data)
            elif option==".tcx":
                tcx_data = api.download_activity(activity_id, dl_fmt=api.ActivityDownloadFormat.TCX)
                with open(os.path.join(out_dir, f"{str(activity_start)}.tcx"), "wb") as fb:
                    fb.write(tcx_data)
            elif option==".zip":
                zip_data = api.download_activity(activity_id, dl_fmt=api.ActivityDownloadFormat.ORIGINAL)
                with open(os.path.join(out_dir, f"{str(activity_start)}.zip"), "wb") as fb:
                    fb.write(zip_data)
            elif option== ".csv":
                csv_data = api.download_activity(activity_id, dl_fmt=api.ActivityDownloadFormat.CSV)
                with open(os.path.join(out_dir, f"{str(activity_start)}.csv"), "wb") as fb:
                    fb.write(csv_data)

######################################################################

def main():
    '''
    navigate into project directory (location of script) & run from there
    file_types: all options = [".tcx", ".gpx", ".csv", ".zip"]
    saves activity files into a new folder named today's date (YYYYMMDD) located in the script/project directory 
    '''
    
    num_days = int(sys.argv[1])
    ## keep python script in the project directory with archive & out folders 
    project_dir=os.getcwd()
    file_types=[".tcx", ".gpx"]
    # Load environment variables if defined
    email = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")
    tokenstore = os.getenv("GARMINTOKENS") or "~/.garminconnect"
    api = None
    
    today = date.today()
    startdate = today - timedelta(days=int(num_days))  
    YYYYMMDD = today.strftime("%Y%m%d")
    out_dir = os.path.join(project_dir, YYYYMMDD)
    print(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    archive_dir = os.path.join(project_dir, "archive")
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    global running_fig_dir
    running_fig_dir = os.path.join(project_dir, "out")
    if not os.path.exists(running_fig_dir):
        os.makedirs(running_fig_dir)        
    
    ## 1) download from garmin
    for option in file_types:
        menu_options = {
            option: f"Download activities data by date from '{startdate.isoformat()}' to '{today.isoformat()}'",
            "q": "Exit" }
        if not api:
            api = init_api(email, password, tokenstore)
        if api:
            switch(api, option, out_dir, startdate, today)
        else:
            api = init_api(email, password, tokenstore)    
    if len(os.listdir(out_dir)) == 0:
        os.rmdir(out_dir)

    return out_dir
    
if __name__ == '__main__':
    main()       