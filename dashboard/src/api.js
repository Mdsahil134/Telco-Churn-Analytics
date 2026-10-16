async function request(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return response.json();
}

export function fetchDashboard() {
  return request("/api/dashboard");
}

export function fetchCustomers(params) {
  const query = new URLSearchParams(
    Object.fromEntries(
      Object.entries(params).filter(([, value]) => value !== "" && value != null)
    )
  );
  return request(`/api/customers?${query.toString()}`);
}

export function fetchPredictions(params) {
  const query = new URLSearchParams(
    Object.fromEntries(
      Object.entries(params).filter(([, value]) => value !== "" && value != null)
    )
  );
  return request(`/api/predictions?${query.toString()}`);
}

export function fetchModel() {
  return request("/api/model");
}
