// script.js

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    loadAnalysisTable();
});

// Main upload and analyze function
document.getElementById('uploadBtn').addEventListener('click', function () {
    const fileInput = document.getElementById('resumeInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Lütfen yüklemek için bir dosya seçin.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('loading').classList.remove('hidden');

    fetch('http://127.0.0.1:5099/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Dosya yüklenemedi.');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }

        const analysisId = data.analysis_id;
        return fetch(`http://127.0.0.1:5099/analysis/${analysisId}`);
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Analiz alınamadı.');
        }
        return response.json();
    })
    .then(analysisData => {
        saveAnalysis(analysisData.metadata.filename, analysisData);
        loadAnalysisTable();
        document.getElementById('loading').classList.add('hidden');
    })
    .catch(error => {
        alert(`Bir hata oluştu: ${error.message}`);
        document.getElementById('loading').classList.add('hidden');
    });
});

// Save analysis data in local storage
function saveAnalysis(filename, analysisData) {
    const analysisStore = JSON.parse(localStorage.getItem('analysisStore')) || {};
    analysisStore[analysisData.metadata.analysis_id] = analysisData;
    localStorage.setItem('analysisStore', JSON.stringify(analysisStore));
}

// Load all analyses into the table
function loadAnalysisTable() {
    const tableBody = document.getElementById('analysisTable').querySelector('tbody');
    tableBody.innerHTML = '';

    const analysisStore = JSON.parse(localStorage.getItem('analysisStore')) || {};
    Object.entries(analysisStore).forEach(([analysisId, analysis]) => {
        const row = document.createElement('tr');
        row.classList.add('border-b');

        row.innerHTML = `
            <td class="py-3 px-6 text-left">${analysisId}</td>
            <td class="py-3 px-6 text-left">${analysis.analysis.contact_info.full_name || 'N/A'}</td>
            <td class="py-3 px-6 text-center">
                <button onclick="viewAnalysis('${analysisId}')" class="bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600 mr-2 transition duration-300 ease-in-out">Detayları Gör</button>
                <button onclick="deleteAnalysis('${analysisId}')" class="bg-red-500 text-white px-3 py-1 rounded hover:bg-red-600 transition duration-300 ease-in-out">Sil</button>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

// View the detailed analysis
function viewAnalysis(analysisId) {
    const analysisStore = JSON.parse(localStorage.getItem('analysisStore')) || {};
    const analysisData = analysisStore[analysisId];

    if (analysisData) {
        displayAnalysis(analysisData);
        document.getElementById('details').classList.remove('hidden');
        // Scroll to the analysis details
        document.getElementById('details').scrollIntoView({ behavior: 'smooth' });
    } else {
        alert('Analiz verisi bulunamadı.');
    }
}

// Display the detailed analysis view with improved layout
function displayAnalysis(analysisData) {
    const output = document.getElementById('analysisOutput');
    output.innerHTML = '';

    const analysis = analysisData.analysis;

    // Create cards for each section
    createInfoCard('İletişim Bilgileri', generateContactInfo(analysis.contact_info), output);
    createInfoCard('Profesyonel Özet', generateProfessionalSummary(analysis.professional_summary), output);
    createInfoCard('Beceriler', generateSkills(analysis.skills), output);

    // Experiences and Education can be in a full-width card
    createFullWidthCard('Deneyimler', generateExperiences(analysis.experiences), output);
    createFullWidthCard('Eğitim', generateEducation(analysis.education), output);

    createInfoCard('Analiz Özeti', generateAnalysisSummary(analysis.analysis_summary), output);

    // Clear previous charts (if any)
    if (window.myCharts) {
        window.myCharts.forEach(chart => chart.destroy());
    }
    window.myCharts = [];

    // Create charts
    if (analysis.metrics && analysis.metrics.profile_strength) {
        createChartCard('Profil Güç Analizi', (canvas) => {
            createRadarChart(analysis.metrics.profile_strength.factors, canvas);
        }, output);
    }

    if (analysis.skills && analysis.skills.technical) {
        const technicalSkills = analysis.skills.technical.map(skill => skill.skills).flat();
        if (technicalSkills.length > 0) {
            createChartCard('Teknik Beceriler Grafiği', (canvas) => {
                createTechnicalSkillsBarChart(technicalSkills, canvas);
            }, output);
        }
    }

    if (analysis.skills && analysis.skills.languages && analysis.skills.languages.length > 0) {
        createChartCard('Dil Yeterliliği', (canvas) => {
            createPolarAreaChart(analysis.skills.languages, canvas);
        }, output);
    }
}

// Helper functions to generate content for cards
function generateContactInfo(contact) {
    return `
        <p><strong>İsim:</strong> ${contact.full_name || 'N/A'}</p>
        <p><strong>E-posta:</strong> ${contact.email || 'N/A'}</p>
        <p><strong>Konum:</strong> ${contact.location || 'N/A'}</p>
        <p><strong>LinkedIn:</strong> ${contact.linkedin || 'N/A'}</p>
    `;
}

function generateProfessionalSummary(summary) {
    return `
        <p><strong>Ünvan:</strong> ${summary.title || 'N/A'}</p>
        <p><strong>Kıdem Seviyesi:</strong> ${summary.seniority_level || 'N/A'}</p>
        <p><strong>Deneyim Yılı:</strong> ${summary.years_of_experience || 'N/A'}</p>
        <p><strong>Sektör Odakları:</strong> ${summary.industry_focus ? summary.industry_focus.join(', ') : 'N/A'}</p>
        <p><strong>Kariyer Öne Çıkanları:</strong> ${summary.career_highlights ? summary.career_highlights.join(', ') : 'N/A'}</p>
    `;
}

function generateSkills(skills) {
    const technicalSkills = skills.technical ? skills.technical.map(skill => skill.skills).flat().join(', ') : 'N/A';
    const softSkills = skills.soft ? skills.soft.join(', ') : 'N/A';
    return `
        <p><strong>Teknik Beceriler:</strong> ${technicalSkills}</p>
        <p><strong>Soft Beceriler:</strong> ${softSkills}</p>
    `;
}

function generateExperiences(experiences) {
    if (!experiences || experiences.length === 0) {
        return '<p>Deneyim bilgisi bulunamadı.</p>';
    }
    return experiences.map(exp => `
        <div class="mb-4">
            <p><strong>Şirket:</strong> ${exp.company.name || 'N/A'}</p>
            <p><strong>Pozisyon:</strong> ${exp.position || 'N/A'}</p>
            <p><strong>Süre:</strong> ${exp.start_date || 'N/A'} - ${exp.end_date || 'N/A'}</p>
            <p><strong>Sorumluluklar:</strong> ${exp.responsibilities ? exp.responsibilities.join(', ') : 'N/A'}</p>
        </div>
    `).join('');
}

function generateEducation(education) {
    if (!education || education.length === 0) {
        return '<p>Eğitim bilgisi bulunamadı.</p>';
    }
    return education.map(edu => `
        <div class="mb-4">
            <p><strong>Kurum:</strong> ${edu.institution || 'N/A'}</p>
            <p><strong>Derece:</strong> ${edu.degree || 'N/A'}</p>
            <p><strong>Alan:</strong> ${edu.field || 'N/A'}</p>
            <p><strong>Süre:</strong> ${edu.start_date || 'N/A'} - ${edu.end_date || 'N/A'}</p>
        </div>
    `).join('');
}

function generateAnalysisSummary(summary) {
    return `
        <p><strong>Güçlü Yönler:</strong> ${summary.strengths ? summary.strengths.join(', ') : 'N/A'}</p>
        <p><strong>Geliştirme Alanları:</strong> ${summary.improvement_areas ? summary.improvement_areas.join(', ') : 'N/A'}</p>
        <p><strong>Benzersiz Satış Noktaları:</strong> ${summary.unique_selling_points ? summary.unique_selling_points.join(', ') : 'N/A'}</p>
        <p><strong>Kariyer Seviyesi Değerlendirmesi:</strong> ${summary.career_level_assessment || 'N/A'}</p>
        <p><strong>Sektör Uyumu:</strong> ${summary.industry_fit ? summary.industry_fit.join(', ') : 'N/A'}</p>
        <p><strong>Sonraki Kariyer Önerileri:</strong> ${summary.next_career_suggestions ? summary.next_career_suggestions.join(', ') : 'N/A'}</p>
    `;
}

// Helper function to create a card
function createInfoCard(title, content, container) {
    const card = document.createElement('div');
    card.className = 'bg-white shadow rounded-lg p-6';
    card.innerHTML = `
        <h3 class="text-xl font-semibold mb-4">${title}</h3>
        ${content}
    `;
    container.appendChild(card);
}

function createFullWidthCard(title, content, container) {
    const cardContainer = document.createElement('div');
    cardContainer.className = 'col-span-1 md:col-span-2';
    const card = document.createElement('div');
    card.className = 'bg-white shadow rounded-lg p-6';
    card.innerHTML = `
        <h3 class="text-xl font-semibold mb-4">${title}</h3>
        ${content}
    `;
    cardContainer.appendChild(card);
    container.appendChild(cardContainer);
}

function createChartCard(title, chartFunction, container) {
    const card = document.createElement('div');
    card.className = 'bg-white shadow rounded-lg p-6';
    const canvasContainer = document.createElement('div');
    canvasContainer.className = 'relative';
    const canvas = document.createElement('canvas');
    canvas.className = 'my-4';
    canvas.style.display = 'block';
    canvas.style.width = '100%';
    canvasContainer.style.width = '100%';
    canvasContainer.style.height = '400px'; // Fixed height for charts
    canvasContainer.appendChild(canvas);
    card.appendChild(canvasContainer);
    const header = document.createElement('h3');
    header.className = 'text-xl font-semibold mb-4';
    header.textContent = title;
    card.insertBefore(header, canvasContainer);
    container.appendChild(card);
    chartFunction(canvas);
}

// Chart creation functions
function createRadarChart(data, canvas) {
    const ctx = canvas.getContext('2d');
    const chart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: Object.keys(data).map(key => key.replace(/_/g, ' ')),
            datasets: [{
                label: 'Skor',
                data: Object.values(data),
                backgroundColor: 'rgba(34, 197, 94, 0.2)',
                borderColor: 'rgba(34, 197, 94, 1)',
                pointBackgroundColor: 'rgba(34, 197, 94, 1)',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, // Allow the chart to fill the container
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    }
                }
            },
            plugins: {
                title: {
                    display: false
                },
                tooltip: {
                    enabled: true
                },
                legend: {
                    display: false
                }
            }
        }
    });
    window.myCharts.push(chart);
}

function createPolarAreaChart(languages, canvas) {
    const ctx = canvas.getContext('2d');
    const chart = new Chart(ctx, {
        type: 'polarArea',
        data: {
            labels: languages.map(lang => lang.language),
            datasets: [{
                data: languages.map(lang => {
                    switch(lang.proficiency) {
                        case 'Native or Bilingual': return 5;
                        case 'Full Professional': return 4;
                        case 'Professional Working': return 3;
                        case 'Limited Working': return 2;
                        case 'Elementary': return 1;
                        default: return 0;
                    }
                }),
                backgroundColor: languages.map(() => getRandomColor()),
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, // Allow the chart to fill the container
            scales: {
                r: {
                    beginAtZero: true,
                    max: 5,
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            plugins: {
                title: {
                    display: false
                },
                legend: {
                    position: 'right'
                },
                tooltip: {
                    enabled: true
                }
            }
        }
    });
    window.myCharts.push(chart);
}

// New function to create technical skills bar chart
function createTechnicalSkillsBarChart(skills, canvas) {
    const skillCounts = {};
    skills.forEach(skill => {
        skillCounts[skill] = (skillCounts[skill] || 0) + 1;
    });

    const labels = Object.keys(skillCounts);
    const data = Object.values(skillCounts);

    const ctx = canvas.getContext('2d');
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Teknik Beceriler Frekansı',
                data: data,
                backgroundColor: labels.map(() => getRandomColor()),
                borderColor: labels.map(() => '#ffffff'),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, // Allow the chart to fill the container
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        autoSkip: false,
                        maxRotation: 90,
                        minRotation: 45
                    }
                },
                y: {
                    beginAtZero: true,
                    stepSize: 1
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: true
                }
            }
        }
    });
    window.myCharts.push(chart);
}

// Helper function to generate random colors
function getRandomColor() {
    const colors = [
        '#F87171', '#FBBF24', '#34D399', '#60A5FA', '#A78BFA',
        '#F472B6', '#FCD34D', '#4ADE80', '#22D3EE', '#818CF8'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
}

// Delete analysis data
function deleteAnalysis(analysisId) {
    const analysisStore = JSON.parse(localStorage.getItem('analysisStore')) || {};
    delete analysisStore[analysisId];
    localStorage.setItem('analysisStore', JSON.stringify(analysisStore));
    loadAnalysisTable();
    // If the deleted analysis was being viewed, close the details
    const details = document.getElementById('details');
    if (!details.classList.contains('hidden')) {
        closeDetails();
    }
}

// Close the detailed analysis view
function closeDetails() {
    document.getElementById('details').classList.add('hidden');
}

// Initialize the analyses table on page load
loadAnalysisTable();
