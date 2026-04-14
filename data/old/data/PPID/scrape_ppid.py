"""
Scraper for PPID UI — saves raw HTML + PDFs, consistent with other scrapers.
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote
import urllib3
import logging
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

OUTPUT_DIR = r"c:\Users\aryan\Documents\kuliah\skripsi\data\PPID"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

CATEGORIES = [
    ("01_DIP_dan_DIK", "https://ppid.ui.ac.id/daftar-informasi-publik-universitas-indonesia/"),
    ("02_peraturan_keputusan_kebijakan", "https://ppid.ui.ac.id/informasi-tentang-peraturan-keputusan-dan-atau-kebijakan-universitas-indonesia/"),
    ("03_rancangan_peraturan", "https://ppid.ui.ac.id/daftar-rancangan-peraturan-universitas-indonesia-2/"),
    ("04_organisasi_administrasi_kepegawaian_keuangan", "https://ppid.ui.ac.id/13524-2/"),
    ("05_perjanjian_pihak_ketiga", "https://ppid.ui.ac.id/surat-surat-perjanjian-dengan-pihak-ketiga/"),
    ("06_surat_menyurat_pimpinan", "https://ppid.ui.ac.id/surat-menyurat-pimpinan-dalam-rangka-pelaksanaan-tugas-pokok-dan-fungsinya/"),
    ("07_perizinan", "https://ppid.ui.ac.id/syarat-syarat-perizinan-izin-yang-diterbitkan-dan-atau-dikeluarkan-berikut-dokumen-pendukungnya-dan-laporan-penataan-izin-yang-diberikan/"),
    ("08_renstra", "https://ppid.ui.ac.id/wp-content/uploads/2024/12/ND-603-Penyampaian-Peraturan-MWA-Nomor-006-Tahun-2024-tentang-Pengesahan-Renstra-UI-2025-2029_Rektor-SA-DGB-2.pdf"),
    ("09_agenda_kerja_pimpinan", "https://ppid.ui.ac.id/agenda-kerja-pimpinan-universitas-indonesia/"),
    ("10_pelanggaran_pengawasan_internal", "https://ppid.ui.ac.id/jumlah-jenis-dan-gambaran-umum-pelanggaran-yang-ditemukan-dalam-pengawasan-internal-serta-laporan-penindakannya/"),
    ("11_pelanggaran_dilaporkan_masyarakat", "https://ppid.ui.ac.id/jumlah-jenis-dan-gambaran-umum-pelanggaran-yang-dilaporkan-oleh-masyarakat-serta-laporan-penindakannya/"),
    ("12_peraturan_perundangan", "https://ppid.ui.ac.id/daftar-peraturan-perundang-undangan-yang-telah-disahkan-beserta-kajian-akademisnya/"),
    ("13_siaran_pers", "https://ppid.ui.ac.id/siaran-pers/"),
    ("14_informasi_terbuka_sengketa", "https://ppid.ui.ac.id/informasi-publik-yang-telah-dinyatakan-terbuka-bagi-masyarakat-berdasarkan-mekanisme-keberatan-dan-atau-penyelesaian-sengketa/"),
    ("15_standar_pengumuman_informasi", "https://ppid.ui.ac.id/standar-pengumuman-informasi/"),
]

# Also scrape the main PPID landing page
EXTRA_PAGES = [
    ("00_informasi_tersedia_setiap_saat", "https://ppid.ui.ac.id/informasi-tersedia-setiap-saat/"),
]


def download_pdf(url, folder_path):
    raw_name = unquote(url.split("/")[-1].split("?")[0])
    filename = re.sub(r"[\r\n\t]", "", raw_name).strip()
    if not filename:
        filename = "document.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    filepath = os.path.join(folder_path, filename)
    if os.path.exists(filepath):
        logging.info(f"    PDF exists: {filename}")
        return True
    try:
        logging.info(f"    Downloading PDF: {filename}")
        r = requests.get(url, verify=False, headers=HEADERS, timeout=20)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        logging.error(f"    Failed: {e}")
        return False


def scrape_page(url, folder_path):
    try:
        r = requests.get(url, verify=False, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        logging.error(f"  Failed to access {url}: {e}")
        return

    # Save raw HTML
    with open(os.path.join(folder_path, "page.html"), "w", encoding="utf-8") as f:
        f.write(r.text)
    logging.info(f"  Saved HTML ({len(r.text)} bytes)")

    # Parse for PDFs (strip sidebar/nav/footer first)
    soup = BeautifulSoup(r.text, "html.parser")
    for unwanted in soup.find_all(["aside", "nav", "footer"]):
        unwanted.decompose()
    for unwanted in soup.find_all(class_=["widget-area", "sidebar", "site-footer"]):
        unwanted.decompose()
    for unwanted in soup.find_all(id=["secondary", "sidebar", "colophon"]):
        unwanted.decompose()

    links = soup.find_all("a", href=True)
    pdf_count = 0
    for a in links:
        full_url = urljoin(url, a["href"]).strip()
        if ".pdf" in full_url.lower():
            if "Pemutakhiran-DIP" in full_url and "01_DIP" not in folder_path:
                continue
            if download_pdf(full_url, folder_path):
                pdf_count += 1
            time.sleep(0.5)

    logging.info(f"  Found {pdf_count} PDFs")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_pages = EXTRA_PAGES + CATEGORIES
    total = len(all_pages)

    for i, (folder_name, url) in enumerate(all_pages, 1):
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        logging.info(f"[{i}/{total}] {folder_name}")

        if url.lower().endswith(".pdf"):
            # Direct PDF link — download it and note it
            download_pdf(url, folder_path)
        else:
            scrape_page(url, folder_path)

        time.sleep(1)


if __name__ == "__main__":
    logging.info("=== Starting PPID scrape ===")
    main()
    logging.info("=== PPID scrape complete ===")
