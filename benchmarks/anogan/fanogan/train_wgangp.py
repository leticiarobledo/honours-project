import os
import torch
import torch.autograd as autograd

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(*real_samples.shape[:2], 1, 1, device=device)

    # interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    fake = torch.ones(*d_interpolates.shape, device=device)

    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgangp(opt, generator, discriminator,
                 dataloader, device, lambda_gp=10):
    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    os.makedirs("results/images", exist_ok=True)

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _)in enumerate(dataloader):
            real_imgs = imgs.to(device)
            optimizer_D.zero_grad()
            z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)
            fake_imgs = generator(z)

            real = discriminator(real_imgs)
            fake = discriminator(fake_imgs.detach())
            gradient_penalty = compute_gradient_penalty(discriminator,
                                                        real_imgs.data,
                                                        fake_imgs.data,
                                                        device)
            # Adversarial loss
            d_loss = (-torch.mean(real) + torch.mean(fake)
                      + lambda_gp * gradient_penalty)

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            if i % opt.n_critic == 0:
                fake_imgs = generator(z)
                g_loss = -torch.mean(discriminator(fake_imgs))
                g_loss.backward()
                optimizer_G.step()
                batches_done += opt.n_critic

    print("saved")
    torch.save(generator.state_dict(), "results/generator")
    torch.save(discriminator.state_dict(), "results/discriminator")
