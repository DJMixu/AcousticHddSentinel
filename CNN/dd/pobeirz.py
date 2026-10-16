import os
import re
import time
import urllib.request
import urllib.error

# Wklej tutaj swój pełny kod HTML z Wayback Machine
html_content = """
!-- WD -->
<br><h1>Western Digital</h1>
<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=wd_loud_clicking.mp3&amp;" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=wd_loud_clicking.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/western_digital">Western Digital</a> desktop drive with <b>bad heads</b> clunking.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=wd_5000gb_slowclicking_spindown.mp3&amp;" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=wd_5000gb_slowclicking_spindown.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/western_digital">Western Digital</a> 500GB desktop drive with <b>bad heads</b> slowly clicks a few times and spins down.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=wd_2500js_slow_click_spin_down.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=wd_2500js_slow_click_spin_down.mp3"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/western_digital">Western Digital</a> 250GB desktop drive with <b>head crash</b> clicks a few times, then spins down.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/sounds/wd_250gb_stuck_spindle_chuttering.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=wd_250gb_stuck_spindle_chuttering.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/western_digital">Western Digital</a> 250GB desktop drive with <b>stuck spindle</b> can't spin up, chatters.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=wd_laptop_clicking.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=wd_laptop_clicking.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/western_digital">Western Digital</a> laptop drive with <b>bad heads</b> making clicking sound.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=wd_200gb_bad_preamp_click_stop.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=wd_200gb_bad_preamp_click_stop.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/western_digital">Western Digital</a> 200GB desktop drive with <b>bad preamplifier chip</b> (located on the headstack) clicks a few times, spins down.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=wd_unstable_clicking_spin_down.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=wd_unstable_clicking_spin_down.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/western_digital">Western Digital</a> desktop drive with <b>unstable heads</b> clicks a few times and stops spinning.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=wd_laptop_stuck_siren.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=wd_laptop_stuck_siren.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/western_digital">Western Digital</a> laptop hard drive with <b>stuck spindle</b> trying to spin up with siren.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=wd_500gb_bad_bearings.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=wd_500gb_bad_bearings.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/western_digital">Western Digital</a> 500GB desktop hard drive with <b>bad bearings</b> can't gain full rotational speed.</p>



<!-- Seagate -->
<br><h1>Seagate</h1>
<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=seagate_dying_heads_spin_down.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=seagate_dying_heads_spin_down.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/seagate">Seagate</a> desktop drive with <b>failing heads</b> making thrashing, then clicking sound.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=seagate_slow_clicking.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=seagate_slow_clicking.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/seagate">Seagate</a> desktop drive with <b>bad heads</b> slowly clicks and beeps on spin up.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=seagate_laptop_clicking.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=seagate_laptop_clicking.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/seagate">Seagate</a> laptop drive with <b>bad heads</b> making clicking/knocking sound.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=seagate_stuck_spindle.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=seagate_stuck_spindle.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/seagate">Seagate</a> desktop drive with <b>seized spindle</b> trying to spin up.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=seagate_momentus_nasty_drilling.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=seagate_momentus_nasty_drilling.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/seagate">Seagate</a> Momentus laptop drive with <b>bad heads</b> making nasty drilling noise.</p>

<!-- MAXTOR -->
<br><h1>Maxtor</h1>
<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=maxtor_calypso_constant_clicking.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=maxtor_calypso_constant_clicking.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/maxtor">Maxtor</a> desktop drive with <b>bad heads</b> making clicking/knocking/beeping noise.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=maxtor_160gb_steady_knocking.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=maxtor_160gb_steady_knocking.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/maxtor">Maxtor</a> drive with <b>bad heads</b> making steady clicking/knocking sound.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=maxtor_melody.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=maxtor_melody.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/maxtor">Maxtor</a> desktop drive with <b>stuck spindle</b> playing futuristic cell phone melody.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=maxtor_melody2.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=maxtor_melody2.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/maxtor">Maxtor</a> drive with <b>stuck spindle</b> and musical siren again.</p>

<!-- SAMSUNG -->
<br><h1>Samsung</h1>
<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=samsung_sp8002_clicking.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=samsung_sp8002_clicking.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/samsung">Samsung</a> 80GB desktop hard drive with <b>bad heads</b> making fast clicking sound.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=samsung_clicks_spins_down.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=samsung_clicks_spins_down.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/samsung">Samsung</a> desktop hard drive with <b>bad heads</b> clicks, then spins down.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=samsung_40gb_bad_heads_knocks.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=samsung_40gb_bad_heads_knocks.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/samsung">Samsung</a> 40GB desktop hard drive with <b>bad head</b> knocks a few times.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=samsung_bad_sectors.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=samsung_bad_sectors.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/samsung">Samsung</a> desktop drive with <b>degraded media</b> making scratching sound when hitting bad sectors.</p>


<!-- HITACHI/IBM -->
<br><h1>Hitachi/IBM</h1>
<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=hitachi_1_click_with_beep.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=hitachi_1_click_with_beep.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/hitachi">Hitachi</a> laptop drive with <b>bad heads</b> clicks once on spin up, then beeps.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=ibm_scratching_bad_sectors.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=ibm_scratching_bad_sectors.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/ibm">IBM</a> desktop drive with <b>degraded media</b> making scratching sound when hitting area with bad sectors.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=hitachi_ibm_clicking.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=hitachi_ibm_clicking.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/hitachi">Hitachi</a>/<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/ibm">IBM</a> laptop drive with <b>bad heads</b> making clicking sound.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=ibm_40gb_bad_heads_rattles_squeals.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=ibm_40gb_bad_heads_rattles_squeals.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/ibm">IBM</a> 40GB desktop hard drive with <b>degraded media/heads</b> rattles and squeals on spin up.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=hitachi_laptop_60gb_nospin_buzzing.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=hitachi_laptop_60gb_nospin_buzzing.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/hitachi">Hitachi</a> 60GB laptop drive with <b>stuck spindle</b> can't spin up, makes humming/buzzing noise.</p>


<!-- TOSHIBA -->
<br><h1>Toshiba</h1>
<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=toshiba_seizure_scratching_no_spin.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=toshiba_seizure_scratching_no_spin.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/toshiba">Toshiba</a> laptop hard drive with <b>stuck spindle</b> trying to spin up(heard if taken close to your ear).</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=toshiba_degrading_bearings.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=toshiba_degrading_bearings.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/toshiba">Toshiba</a> laptop drive with <b>failing bearings</b> making grinding sound.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=toshiba_60gb_loud_grinding.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=toshiba_60gb_loud_grinding.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/toshiba">Toshiba</a> laptop drive with <b>bad bearings</b> making loud grinding sound.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=toshiba_60gb_loud_drilling.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=toshiba_60gb_loud_drilling.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/toshiba">Toshiba</a> laptop drive with <b>bad bearings</b> making nasty drilling/screaming sound.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=toshiba_clicking_sweeping.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=toshiba_clicking_sweeping.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/toshiba">Toshiba</a> laptop drive with <b>bad heads</b> making clicking/sweeping sound on boot up.</p>

<!-- FUJITSU -->
<br><h1>Fujitsu</h1>
<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=fujitsu_60gb_clicking.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=fujitsu_60gb_clicking.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/fujitsu">Fujitsu</a> laptop drive with <b>bad heads</b> making clicking/knocking noise.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=fujitsu_60gb_sweeping.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=fujitsu_60gb_sweeping.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/fujitsu">Fujitsu</a> laptop hard drive with <b>bad heads</b> making sweeping sound.</p>

<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=fujitsu_20gb_bad_media_scratches.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=fujitsu_20gb_bad_media_scratches.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/fujitsu">Fujitsu</a> 40gb desktop drive with <b>bad media</b> making scratching noise.</p>


<!-- QUANTUM -->
<br><h1>Quantum</h1>
<p><object type="application/x-shockwave-flash" data="/web/20200216022247im_/http://datacent.com/musicplayer.swf?song_url=/getsound.php?file=quantum_clunking.mp3" width="20" height="17"><param name="movie" value="/musicplayer.swf?song_url=/getsound.php?file=quantum_clunking.mp3&amp;"></object>
<a href="/web/20200216022247/http://datacent.com/datarecovery/hdd/quantum">Quantum</a> desktop drive with <b>bad heads</b> making clunking sound.</p>

"""

pattern = r'song_url=([^&"\']+\.mp3)'
matches = re.findall(pattern, html_content)

base_wayback = "https://web.archive.org/web/20200216022247"
download_links = set()

for match in matches:
    clean_match = match.strip()
    if clean_match.startswith('http'):
        parts = clean_match.split('http', 1)
        fixed_link = f"{base_wayback}id_/http{parts[1]}"
    elif clean_match.startswith('/'):
        fixed_link = f"{base_wayback}id_{clean_match}"
    else:
        fixed_link = f"{base_wayback}id_/http://datacent.com/getsound.php?file={clean_match.split('/')[-1]}"

    download_links.add(fixed_link)

output_dir = "hdd_sounds"
os.makedirs(output_dir, exist_ok=True)

links_list = sorted(list(download_links))
total = len(links_list)

print(f"Znaleziono {total} plików. Uruchamiam wolne pobieranie z odpornością na bana...\n")

for idx, link in enumerate(links_list, 1):
    file_name = link.split("=")[-1].split("/")[-1].split("?")[0]
    file_path = os.path.join(output_dir, file_name)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 2048:
        print(f"[{idx}/{total}] Pomijam: {file_name} (już pobrany)")
        continue

    print(f"[{idx}/{total}] Pobieranie: {file_name} ...")

    success = False
    attempts = 0
    while not success and attempts < 5:
        try:
            # Udajemy prawdziwą przeglądarkę z pełnymi nagłówkami
            req = urllib.request.Request(
                link,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                    'Accept': 'audio/webm,audio/ogg,audio/wav,audio/*;q=0.9,application/rdf+xml;q=0.8,*/*;q=0.5',
                    'Accept-Language': 'pl,en-US;q=0.7,en;q=0.3',
                    'Connection': 'keep-alive'
                }
            )
            with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                out_file.write(response.read())

            if os.path.getsize(file_path) > 2048:
                print(f"  -> Sukces!")
                success = True
            else:
                print(f"  -> Plik za mały, czekam i ponawiam...")
                attempts += 1
                time.sleep(15)

        except urllib.error.HTTPError as e:
            if e.code == 429:
                attempts += 1
                # Jeśli dostaniesz bana, skrypt poczeka coraz dłużej: 30s, 60s, 90s...
                wait_time = attempts * 30
                print(f"  -> Blokada 429 (Too Many Requests). Czekam {wait_time} sekund na odblokowanie IP...")
                time.sleep(wait_time)
            else:
                print(f"  -> Błąd HTTP {e.code}: {e.reason}")
                break
        except Exception as e:
            print(f"  -> Błąd: {e}")
            break

    # Bardzo bezpieczna przerwa 10 sekund między plikami
    if idx < total:
        time.sleep(10)

print("\nProces zakończony!")
