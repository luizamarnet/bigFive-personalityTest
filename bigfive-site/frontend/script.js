const questions = [
  { id: "EXT1", text: "I am the life of the party." },
  { id: "EXT2", text: "I don't talk a lot." },
  { id: "EXT3", text: "I feel comfortable around people." },
  { id: "EXT4", text: "I keep in the background." },
  { id: "EXT5", text: "I start conversations." },
  { id: "EXT6", text: "I have little to say." },
  { id: "EXT7", text: "I talk to a lot of different people at parties." },
  { id: "EXT8", text: "I don't like to draw attention to myself." },
  { id: "EXT9", text: "I don't mind being the center of attention." },
  { id: "EXT10", text: "I am quiet around strangers." },
  { id: "EST1", text: "I get stressed out easily." },
  { id: "EST2", text: "I am relaxed most of the time." },
  { id: "EST3", text: "I worry about things." },
  { id: "EST4", text: "I seldom feel blue." },
  { id: "EST5", text: "I am easily disturbed." },
  { id: "EST6", text: "I get upset easily." },
  { id: "EST7", text: "I change my mood a lot." },
  { id: "EST8", text: "I have frequent mood swings." },
  { id: "EST9", text: "I get irritated easily." },
  { id: "EST10", text: "I often feel blue." },
  { id: "AGR1", text: "I feel little concern for others." },
  { id: "AGR2", text: "I am interested in people." },
  { id: "AGR3", text: "I insult people." },
  { id: "AGR4", text: "I sympathize with others' feelings." },
  { id: "AGR5", text: "I am not interested in other people's problems." },
  { id: "AGR6", text: "I have a soft heart." },
  { id: "AGR7", text: "I am not really interested in others." },
  { id: "AGR8", text: "I take time out for others." },
  { id: "AGR9", text: "I feel others' emotions." },
  { id: "AGR10", text: "I make people feel at ease." },
  { id: "CSN1", text: "I am always prepared." },
  { id: "CSN2", text: "I leave my belongings around." },
  { id: "CSN3", text: "I pay attention to details." },
  { id: "CSN4", text: "I make a mess of things." },
  { id: "CSN5", text: "I get chores done right away." },
  { id: "CSN6", text: "I often forget to put things back in their proper place." },
  { id: "CSN7", text: "I like order." },
  { id: "CSN8", text: "I shirk my duties." },
  { id: "CSN9", text: "I follow a schedule." },
  { id: "CSN10", text: "I am exacting in my work." },
  { id: "OPN1", text: "I have a rich vocabulary." },
  { id: "OPN2", text: "I have difficulty understanding abstract ideas." },
  { id: "OPN3", text: "I have a vivid imagination." },
  { id: "OPN4", text: "I am not interested in abstract ideas." },
  { id: "OPN5", text: "I have excellent ideas." },
  { id: "OPN6", text: "I do not have a good imagination." },
  { id: "OPN7", text: "I am quick to understand things." },
  { id: "OPN8", text: "I use difficult words." },
  { id: "OPN9", text: "I spend time reflecting on things." },
  { id: "OPN10", text: "I am full of ideas." },
];

const container = document.getElementById("questions");

questions.forEach((q) => {
  const block = document.createElement("div");
  block.classList.add("question-block");
  block.innerHTML = `
    <p><strong>${q.text}</strong></p>
    <div class="options">
      ${[1, 2, 3, 4, 5]
        .map(
          (val) => `
        <label>
          <input type="radio" name="${q.id}" value="${val}" required />
          ${val}
        </label>`
        )
        .join("")}
    </div>
    <hr/>
  `;
  container.appendChild(block);
});

document.getElementById("bigfive-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const answers = {};
  for (const [key, value] of formData.entries()) {
    answers[key] = parseInt(value);
  }

  if (Object.keys(answers).length !== 50) {
    alert("Por favor, responda todas as perguntas.");
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:8000/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answers }),
    });

    const result = await response.json();

    const resultEl = document.getElementById("result");
    resultEl.innerHTML = "<h2>Resultado do Modelo:</h2>";
    for (const [trait, score] of Object.entries(result.scores)) {
      resultEl.innerHTML += `<p><strong>${trait}:</strong> ${score}</p>`;
    }
    console.log("Scores recebidos:", result.scores);
    desenharGraficoRadar(result.scores);
    
  } catch (err) {
    alert("Erro ao enviar respostas.");
    console.error(err);
  }
});


function desenharGraficoRadar(scores) {
  const ctx = document.getElementById("radarChart").getContext("2d");

  // Destroi gráfico antigo se já existir
  if (window.radarInstance) {
    window.radarInstance.destroy();
  }

  window.radarInstance = new Chart(ctx, {
    type: "radar",
    data: {
      labels: ["Extroversão", "Neuroticismo", "Agradabilidade", "Conscienciosidade", "Abertura"],
      datasets: [{
        label: "Perfil de Personalidade",
        data: [
          scores["Extraversion"],//["Extroversão"],
          scores["Neuroticism"],//["Neuroticismo"],
          scores["Agreeableness"],//["Agradabilidade"],
          scores["Conscientiousness"],//["Conscienciosidade"],
          scores["Openness"]//["Abertura"]
        ],
        backgroundColor: "rgba(102, 51, 153, 0.2)",
        borderColor: "rgba(102, 51, 153, 1)",
        borderWidth: 2,
        pointBackgroundColor: "rgba(102, 51, 153, 1)"
      }]
    },
    options: {
      scales: {
        r: {
          suggestedMin: 0,
          suggestedMax: 1,
          ticks: {
            stepSize: 0.2
          }
        }
      },
      responsive: true,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: "Gráfico Radar – Big Five",
          font: { size: 16 }
        }
      }
    }
  });
}
